"""Orchestrates the sequential continual-learning update procedure.

Given an initial snapshot ``T_0`` and a list of
``dicee.continual.update_sequence.UpdateStep``, :class:`ContinualTrainer`:

1. Trains (or loads) a ``BaseKGE`` model on ``T_0`` (initial training is a
   thin wrapper that reuses the model's own ``forward_k_vs_all`` /
   ``loss_function`` exactly as the rest of DICE does -- see
   ``_train_kvsall_epochs``).
2. For every subsequent step ``i``:
   a. Freezes a snapshot of the current model (``model_old``, no grad).
   b. Grows the entity embedding table if the step introduces new
      entities (:func:`dicee.continual.initialization.extend_entity_embeddings`)
      and rebuilds the optimizer (new parameter set -> optimizer state
      must be rebuilt, see ``dicee/continual/initialization.py``).
   c. Computes Chebyshev influence scores for the step (unless running the
      plain fine-tuning or uniform-distillation baselines).
   d. Runs a few epochs of combined adaptation + (optional) preservation
      training on ``DeltaT_i`` (adaptation) and a stable-query sample
      (preservation).
   e. Evaluates the resulting snapshot and preservation metrics.

Three methods share the exact same update sequence and initial checkpoint:

* ``"finetune"``      -- ``lambda_pres = 0`` (no preservation term).
* ``"uniform"``       -- preservation with ``w(q) = 1`` for every stable query.
* ``"chebyshev"``     -- preservation with Chebyshev-derived ``w(q)``.

This module intentionally does **not** go through ``dicee.executer.Execute``
/ ``dicee.trainer.dice_trainer.DICE_Trainer``: those orchestrate a single,
static training run (one call to ``PreprocessKG`` + one trainer backend) and
assume a fixed entity vocabulary for the whole run, which is incompatible
with vocabulary growth across update steps. Everything *inside* each step,
however (the model's forward/loss, the evaluation ranker, ...), reuses the
existing DICE APIs directly.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import torch

from dicee.static_preprocess_funcs import get_er_vocab, get_re_vocab

from .evaluation import evaluate_snapshot, preservation_metrics
from .continual_losses import (
    adaptation_loss,
    margin_adaptation_loss,
    preservation_loss,
    preservation_weights,
    sample_filtered_negatives,
)
from .influence import compute_influence_scores
from .initialization import extend_entity_embeddings
from .update_sequence import UpdateStep

logger = logging.getLogger(__name__)

Method = Literal["finetune", "uniform", "chebyshev"]


@dataclass
class ContinualConfig:
    """Hyperparameters for one continual-learning run (Section 12)."""

    method: Method = "finetune"
    lambda_pres: float = 1.0
    beta: float = 2.0
    cheb_K: int = 3
    cheb_rho: float = 0.9
    cheb_epsilon: float = 1e-8
    epochs_per_step: int = 5
    lr: Optional[float] = None
    num_stable_queries: int = 256
    seed: int = 0
    margin: float = 1.0
    num_negatives: int = 1


def _build_er_re_vocab(triples: np.ndarray):
    pairs = [tuple(int(x) for x in row) for row in triples]
    return get_er_vocab(pairs), get_re_vocab(pairs)


def _train_epochs_kvsall(model, triples: np.ndarray, num_entities: int, epochs: int,
                          optimizer: torch.optim.Optimizer, batch_size: int = 128) -> List[float]:
    """Train ``model`` for ``epochs`` epochs on ``triples`` via KvsAll.

    Builds standard multi-hot KvsAll targets from ``get_er_vocab`` (exactly
    the labelling DICE's own ``KvsAll`` dataset class uses) and reuses
    ``model.forward_k_vs_all`` + ``model.loss_function`` for the adaptation
    loss -- no scoring/loss logic is reimplemented.
    """
    if len(triples) == 0:
        return []
    er_vocab = get_er_vocab([tuple(int(x) for x in row) for row in triples])
    keys = list(er_vocab.keys())
    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for start in range(0, len(keys), batch_size):
            batch_keys = keys[start:start + batch_size]
            x = torch.tensor(batch_keys, dtype=torch.long)
            y = torch.zeros(len(batch_keys), num_entities, dtype=torch.float32)
            for row, key in enumerate(batch_keys):
                y[row, er_vocab[key]] = 1.0

            optimizer.zero_grad()
            yhat = model.forward_k_vs_all(x)
            loss = adaptation_loss(model, yhat, y, current_epoch=epoch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / max(1, len(keys) // batch_size + 1))
    return losses


class ContinualTrainer:
    """Coordinates one (model, update-sequence) continual-learning run."""

    def __init__(self, model: torch.nn.Module, config: ContinualConfig):
        self.model = model
        self.config = config
        self.history: List[Dict] = []

    def _make_optimizer(self) -> torch.optim.Optimizer:
        lr = self.config.lr or self.model.args.get("lr", 0.01)
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_initial(self, t0_triples: np.ndarray) -> Dict:
        optimizer = self._make_optimizer()
        losses = _train_epochs_kvsall(
            self.model, t0_triples, self.model.num_entities,
            self.config.epochs_per_step, optimizer,
        )
        self.optimizer = optimizer
        return {"step": 0, "losses": losses, "num_triples": len(t0_triples)}

    def run_step(
        self,
        step: UpdateStep,
        new_num_entities: int,
        stable_query_pool: np.ndarray,
        rng: np.random.Generator,
    ) -> Dict:
        """Execute one continual-learning update step and return its metrics.

        ``new_num_entities`` must be large enough to index *every* entity id
        referenced in ``step.cumulative_triples`` (typically
        ``int(step.cumulative_triples[:, [0, 2]].max()) + 1``, or simply the
        full dataset vocabulary size if entity ids are assigned once upfront
        from the complete ``entity_to_idx`` mapping, as is standard in DICE).
        """
        t_start = time.time()

        # (a) freeze a snapshot of the model BEFORE this step's updates.
        model_old = copy.deepcopy(self.model)
        for p in model_old.parameters():
            p.requires_grad_(False)
        model_old.eval()

        # (b) grow the entity table if needed, then rebuild the optimizer
        # (Adam state is tied to parameter identity; a fresh optimizer is
        # required whenever entity_embeddings is replaced, see
        # dicee/continual/initialization.py).
        if new_num_entities > self.model.num_entities:
            extend_entity_embeddings(self.model, new_num_entities)
            optimizer = self._make_optimizer()
        else:
            optimizer = self.optimizer
        # The model may already cover more entities than `new_num_entities`
        # requests (e.g. a fixed full-vocabulary table set up from the
        # start); always use the model's *actual*, current entity count as
        # the authoritative width for KvsAll targets / propagation graphs.
        num_entities_now = self.model.num_entities

        # (c) influence scores (only needed for the Chebyshev method).
        influence_hat = {}
        if self.config.method == "chebyshev":
            influence_hat = compute_influence_scores(
                step.cumulative_triples, step.added_triples, step.new_entity_ids,
                num_entities_now, K=self.config.cheb_K, rho=self.config.cheb_rho,
                epsilon=self.config.cheb_epsilon,
            )

        # (d) stable-query sample for this step (exclude current update batch).
        added_set = {tuple(int(x) for x in row) for row in step.added_triples}
        candidates = [tuple(int(x) for x in row) for row in stable_query_pool
                      if tuple(int(x) for x in row) not in added_set]
        n_sample = min(self.config.num_stable_queries, len(candidates))
        if n_sample > 0:
            sel_idx = rng.choice(len(candidates), size=n_sample, replace=False)
            stable_batch = np.array([candidates[i] for i in sel_idx], dtype=np.int64)
        else:
            stable_batch = np.zeros((0, 3), dtype=np.int64)

        # Known positives at the *target* snapshot (T_{i+1}): used to filter
        # negative sampling so true-but-unseen facts (including ones
        # involving brand-new entities) are never used as negative
        # supervision. Deliberately built from `cumulative_triples`, not just
        # `added_triples` -- see `sample_filtered_negatives` docstring.
        known_positive_set = {tuple(int(x) for x in row) for row in step.cumulative_triples}
        pos_triples_np = np.asarray(step.added_triples, dtype=np.int64)

        losses = []
        self.model.train()
        for epoch in range(self.config.epochs_per_step):
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, max(1, len(pos_triples_np)), 128):
                batch_pos = pos_triples_np[start:start + 128]
                optimizer.zero_grad()
                total = torch.tensor(0.0)
                if len(batch_pos) > 0:
                    neg_np = sample_filtered_negatives(
                        batch_pos, num_entities_now, known_positive_set,
                        self.config.num_negatives, rng,
                    )
                    pos_t = torch.tensor(batch_pos, dtype=torch.long)
                    neg_t = torch.tensor(neg_np, dtype=torch.long)
                    total = margin_adaptation_loss(self.model, pos_t, neg_t, margin=self.config.margin)

                if self.config.method != "finetune" and len(stable_batch) > 0:
                    h = torch.tensor(stable_batch[:, 0], dtype=torch.long)
                    r = torch.tensor(stable_batch[:, 1], dtype=torch.long)
                    t = torch.tensor(stable_batch[:, 2], dtype=torch.long)
                    if self.config.method == "uniform":
                        weights = torch.ones(len(stable_batch))
                    else:
                        weights = preservation_weights(h, t, influence_hat, self.config.beta, h.device)
                    pres = preservation_loss(self.model, model_old, h, r, t, weights)
                    total = total + self.config.lambda_pres * pres

                if total.requires_grad:
                    total.backward()
                    optimizer.step()
                epoch_loss += float(total.detach())
                n_batches += 1
            losses.append(epoch_loss / max(1, n_batches))

        self.optimizer = optimizer

        result = {
            "step": step.step,
            "method": self.config.method,
            "losses": losses,
            "num_added_triples": step.num_added_triples,
            "num_new_entities": step.num_new_entities,
            "training_time_seconds": time.time() - t_start,
            "num_trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        self.history.append(result)
        self._last_model_old = model_old
        return result
