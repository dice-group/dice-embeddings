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
from tqdm.auto import tqdm

from dicee.static_preprocess_funcs import get_er_vocab, get_re_vocab

from .evaluation import evaluate_snapshot, preservation_metrics
from .continual_losses import (
    adaptation_loss,
    encode_triples_as_keys,
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
    num_negatives: int = 10
    device: str = "cpu"
    kvsall_batch_size: int = 1024
    step_batch_size: int = 1024
    # IMPORTANT (root-cause fix, see run_step below): default is False.
    # A fixed-cap `torch.renorm(..., maxnorm=sqrt(dim))` was previously
    # applied unconditionally after every optimizer step during continual
    # updates (but NEVER during `train_initial`, i.e. never for T0 or for
    # the retrain/oracle baseline). Measured on the real FB_CKGE T0
    # DistMult(dim=32) checkpoint, ~32% of entity rows already exceed
    # sqrt(32)=5.657 after ordinary KvsAll/BCE training (mean row norm
    # 5.32, max 15.06), with individual rows losing up to ~62% of their
    # norm in a *single* renorm call -- repeated hundreds of times per
    # update step. This is a large, asymmetric (continual-only)
    # perturbation of an already-converged representation and was
    # confirmed to be the dominant cause of the FB_CKGE MRR collapse
    # (0.18 retrain vs. ~0.02 chebyshev/finetune/uniform). It is kept as
    # an opt-in safety valve (set True) for genuinely unstable runs, but
    # must never be the default.
    normalize_embeddings: bool = False
    grad_clip_norm: Optional[float] = 1.0
    embedding_max_norm: Optional[float] = None
    # Early stopping (FastKGE/IncDE-style): used only by `train_initial`
    # (T0 training / retrain-from-scratch oracle). If set, `epochs_per_step`
    # (acting as a max-epoch cap) is combined with periodic validation-MRR
    # checks; training stops once `early_stopping_patience` consecutive
    # epochs fail to improve val MRR by at least `early_stopping_min_delta`,
    # and the best-val-MRR epoch's weights are restored at the end. None
    # (default) disables early stopping entirely (train for exactly
    # `epochs_per_step` epochs, as before).
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 1e-4
    # IncDE-style two-stage training: for the first `two_stage_epochs`
    # epochs of EACH continual step, gradients for entity rows that already
    # existed before this step are zeroed out before every optimizer.step()
    # call, so only newly-introduced entity rows (plus all relation/other
    # parameters) are updated. After `two_stage_epochs`, training proceeds
    # as normal (joint fine-tuning of old+new rows). 0 (default) disables
    # this and trains all rows from the first epoch, as before.
    two_stage_epochs: int = 0


def _build_er_re_vocab(triples: np.ndarray):
    pairs = [tuple(int(x) for x in row) for row in triples]
    return get_er_vocab(pairs), get_re_vocab(pairs)



def _train_epochs_kvsall(model, triples: np.ndarray, num_entities: int, epochs: int,
                          optimizer: torch.optim.Optimizer, batch_size: int = 1024,
                          device: Optional[torch.device] = None, desc: str = "T0 training",
                          val_triples: Optional[np.ndarray] = None, val_er_vocab: Optional[Dict] = None,
                          val_re_vocab: Optional[Dict] = None, patience: Optional[int] = None,
                          min_delta: float = 1e-4) -> List[float]:
    """Train ``model`` for up to ``epochs`` epochs on ``triples`` via KvsAll.

    If ``val_triples``/``val_er_vocab``/``val_re_vocab``/``patience`` are all
    given, this additionally evaluates validation MRR after every epoch and
    stops early once ``patience`` consecutive epochs fail to improve it by
    at least ``min_delta`` (FastKGE/IncDE-style patience-based early
    stopping), restoring the best-val-MRR epoch's weights before returning.
    ``epochs`` then acts as a maximum-epoch cap rather than a fixed count.

    Builds standard multi-hot KvsAll targets from ``get_er_vocab`` (exactly
    the labelling DICE's own ``KvsAll`` dataset class uses) and reuses
    ``model.forward_k_vs_all`` + ``model.loss_function`` for the adaptation
    loss -- no scoring/loss logic is reimplemented.

    The ``(h, r) -> {tails}`` -> multi-hot target scatter is precomputed
    ONCE per batch (not re-flattened every epoch), and the dense target
    tensor is filled via a single vectorized ``y[rows, cols] = 1.0`` index
    assignment instead of a Python-level per-row loop -- this matters a lot
    at scale (e.g. FB_CKGE's T0 snapshot has >80k unique ``(h, r)`` keys),
    where a per-row Python loop re-executed every epoch dominates wall-clock
    time well before the GPU matmul itself becomes the bottleneck.
    """
    if len(triples) == 0:
        return []
    device = device or next(model.parameters()).device
    er_vocab = get_er_vocab([tuple(int(x) for x in row) for row in triples])
    keys = list(er_vocab.keys())

    # Precompute every batch's (x, scatter_rows, scatter_cols) once; reused
    # identically across every epoch (the labelling never changes).
    batches = []
    for start in range(0, len(keys), batch_size):
        batch_keys = keys[start:start + batch_size]
        x = torch.tensor(batch_keys, dtype=torch.long, device=device)
        rows: List[int] = []
        cols: List[int] = []
        for i, key in enumerate(batch_keys):
            tails = er_vocab[key]
            rows.extend([i] * len(tails))
            cols.extend(tails)
        row_idx = torch.tensor(rows, dtype=torch.long, device=device)
        col_idx = torch.tensor(cols, dtype=torch.long, device=device)
        batches.append((x, row_idx, col_idx, len(batch_keys)))

    losses = []
    model.train()
    do_early_stopping = (
        patience is not None and val_triples is not None
        and val_er_vocab is not None and val_re_vocab is not None and len(val_triples) > 0
    )
    best_val_mrr = -1.0
    best_state = None
    epochs_no_improve = 0

    pbar = tqdm(range(epochs), desc=desc, unit="epoch", leave=False)
    for epoch in pbar:
        epoch_loss = 0.0
        for x, row_idx, col_idx, b in batches:
            y = torch.zeros(b, num_entities, dtype=torch.float32, device=device)
            y[row_idx, col_idx] = 1.0

            optimizer.zero_grad()
            yhat = model.forward_k_vs_all(x)
            loss = adaptation_loss(model, yhat, y, current_epoch=epoch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(1, len(batches))
        losses.append(avg_loss)

        if do_early_stopping:
            model.eval()
            with torch.no_grad():
                val_metrics = evaluate_snapshot(model, val_triples, val_er_vocab, val_re_vocab)
            model.train()
            val_mrr = val_metrics["MRR"]
            pbar.set_postfix(loss=f"{avg_loss:.4f}", val_MRR=f"{val_mrr:.4f}")
            if val_mrr > best_val_mrr + min_delta:
                best_val_mrr = val_mrr
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        "%s: early stopping at epoch %s/%s (best val MRR=%.4f, no improvement "
                        "for %s epochs).", desc, epoch + 1, epochs, best_val_mrr, patience,
                    )
                    break
        else:
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

    if do_early_stopping and best_state is not None:
        model.load_state_dict(best_state)
    return losses




class ContinualTrainer:
    """Coordinates one (model, update-sequence) continual-learning run."""

    def __init__(self, model: torch.nn.Module, config: ContinualConfig):
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.config = config
        self.history: List[Dict] = []

    def _make_optimizer(self) -> torch.optim.Optimizer:
        lr = self.config.lr or self.model.args.get("lr", 0.01)
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_initial(self, t0_triples: np.ndarray, val_triples: Optional[np.ndarray] = None,
                       val_er_vocab: Optional[Dict] = None, val_re_vocab: Optional[Dict] = None) -> Dict:
        optimizer = self._make_optimizer()
        losses = _train_epochs_kvsall(
            self.model, t0_triples, self.model.num_entities,
            self.config.epochs_per_step, optimizer, batch_size=self.config.kvsall_batch_size, device=self.device,
            desc=f"[{self.config.method}] T0 training",
            val_triples=val_triples, val_er_vocab=val_er_vocab, val_re_vocab=val_re_vocab,
            patience=self.config.early_stopping_patience, min_delta=self.config.early_stopping_min_delta,
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
        old_num_entities = self.model.num_entities

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
            old_rows_before = self.model.entity_embeddings.weight.data[:old_num_entities].clone()
            extend_entity_embeddings(self.model, new_num_entities)
            self.model = self.model.to(self.device)
            optimizer = self._make_optimizer()
            # Diagnostics (Section 3 of the audit): verify old rows survived
            # the table swap bit-identical and the table/optimizer now cover
            # exactly the requested vocabulary.
            assert self.model.num_entities == new_num_entities, (
                f"entity table did not grow to the requested size: "
                f"{self.model.num_entities} != {new_num_entities}"
            )
            new_rows_after = self.model.entity_embeddings.weight.data[:old_num_entities].to(old_rows_before.device)
            assert torch.allclose(old_rows_before, new_rows_after), (
                "extend_entity_embeddings mutated pre-existing entity rows "
                "(expected bit-identical old rows after table growth)."
            )
            num_trainable_emb = sum(
                p.numel() for n, p in self.model.named_parameters()
                if n == "entity_embeddings.weight" and p.requires_grad
            )
            assert num_trainable_emb == new_num_entities * self.model.embedding_dim, (
                "entity_embeddings.weight is not fully trainable after extension "
                f"({num_trainable_emb} trainable scalars, expected "
                f"{new_num_entities * self.model.embedding_dim})."
            )
            logger.info(
                "[step %s] entities %s -> %s (+%s new); %s old rows preserved; "
                "%s trainable embedding scalars.",
                step.step, old_num_entities, new_num_entities,
                new_num_entities - old_num_entities, old_num_entities, num_trainable_emb,
            )
        else:
            optimizer = self.optimizer
        # The model may already cover more entities than `new_num_entities`
        # requests (e.g. a fixed full-vocabulary table set up from the
        # start); always use the model's *actual*, current entity count as
        # the authoritative width for KvsAll targets / propagation graphs.
        num_entities_now = self.model.num_entities

        # Entity-id range diagnostics: every id referenced by this step's
        # training data must be a valid row index into the (possibly just
        # grown) embedding table.
        if len(step.cumulative_triples):
            ent_ids = step.cumulative_triples[:, [0, 2]]
            min_id, max_id = int(ent_ids.min()), int(ent_ids.max())
            n_out_of_range = int(((ent_ids < 0) | (ent_ids >= num_entities_now)).sum())
            assert n_out_of_range == 0, (
                f"[step {step.step}] {n_out_of_range} entity ids in cumulative_triples "
                f"fall outside [0, {num_entities_now}); min={min_id}, max={max_id}."
            )
            logger.debug(
                "[step %s] entity id range [%s, %s] within valid table size %s.",
                step.step, min_id, max_id, num_entities_now,
            )

        # (c) influence scores (only needed for the Chebyshev method).
        influence_hat = {}
        if self.config.method == "chebyshev":
            influence_hat = compute_influence_scores(
                step.cumulative_triples, step.added_triples, step.new_entity_ids,
                num_entities_now, K=self.config.cheb_K, rho=self.config.cheb_rho,
                epsilon=self.config.cheb_epsilon,
            )

            # print(f"[step {step.step}] influence scores: {influence_hat}")

        # (d) stable-query sample for this step (exclude current update batch).
        added_set = {tuple(int(x) for x in row) for row in step.added_triples}
        candidates = [tuple(int(x) for x in row) for row in  stable_query_pool
                      if tuple(int(x) for x in row) not in added_set]
        n_sample = min(self.config.num_stable_queries, len(candidates))
        if n_sample > 0:
            sel_idx = rng.choice(len(candidates), size=n_sample, replace=False)
            stable_batch = np.array([candidates[i] for i in sel_idx], dtype=np.int64)
        else:
            stable_batch = np.zeros((0, 3), dtype=np.int64)

        known_positive_keys = encode_triples_as_keys(
            step.cumulative_triples, num_entities_now, self.model.num_relations
        )
        pos_triples_np = np.asarray(step.added_triples, dtype=np.int64)

        losses = []
        self.model.train()
        pbar = tqdm(range(self.config.epochs_per_step), desc=f"[{self.config.method}] step {step.step}",
                    unit="epoch", leave=False)
        for epoch in pbar:
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, max(1, len(pos_triples_np)), self.config.step_batch_size):
                batch_pos = pos_triples_np[start:start + self.config.step_batch_size]
                optimizer.zero_grad()
                total = torch.tensor(0.0, device=self.device)
                if len(batch_pos) > 0:
                    neg_np = sample_filtered_negatives(
                        batch_pos, num_entities_now, known_positive_keys,
                        self.config.num_negatives, rng, num_relations=self.model.num_relations,
                    )
                    pos_t = torch.tensor(batch_pos, dtype=torch.long, device=self.device)
                    neg_t = torch.tensor(neg_np, dtype=torch.long, device=self.device)
                    total = margin_adaptation_loss(self.model, pos_t, neg_t, margin=self.config.margin)

                if self.config.method != "finetune" and len(stable_batch) > 0:
                    h = torch.tensor(stable_batch[:, 0], dtype=torch.long, device=self.device)
                    r = torch.tensor(stable_batch[:, 1], dtype=torch.long, device=self.device)
                    t = torch.tensor(stable_batch[:, 2], dtype=torch.long, device=self.device)
                    if self.config.method == "uniform":
                        weights = torch.ones(len(stable_batch), device=self.device)
                    else:
                        weights = preservation_weights(h, t, influence_hat, self.config.beta, h.device)
                    pres = preservation_loss(self.model, model_old, h, r, t, weights)
                    total = total + self.config.lambda_pres * pres

                if total.requires_grad:
                    total.backward()
                    if self.config.two_stage_epochs > 0 and epoch < self.config.two_stage_epochs and old_num_entities > 0:
                        # IncDE-style two-stage training: during the first
                        # `two_stage_epochs` epochs of this step, only newly
                        # introduced entity rows are allowed to update;
                        # pre-existing rows' gradients are zeroed out here so
                        # `optimizer.step()` leaves them untouched. Joint
                        # fine-tuning of old+new rows resumes afterwards.
                        grad = self.model.entity_embeddings.weight.grad
                        if grad is not None:
                            grad[:old_num_entities] = 0.0
                    if self.config.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                    optimizer.step()
                    if self.config.normalize_embeddings:
                       
                        with torch.no_grad():
                            w = self.model.entity_embeddings.weight.data
                            if self.config.embedding_max_norm is not None:
                                max_norm = self.config.embedding_max_norm
                            else:
                                max_norm = 4.0 * float(w.norm(dim=1).max().item())
                            w.copy_(torch.renorm(w, p=2, dim=0, maxnorm=max_norm))
                epoch_loss += float(total.detach())
                n_batches += 1
            avg_loss = epoch_loss / max(1, n_batches)
            losses.append(avg_loss)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

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
