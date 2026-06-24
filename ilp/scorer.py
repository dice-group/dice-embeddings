"""`Scorer`: an ergonomic, efficient facade for scoring with a trained model.

`InductiveKGModel.forward` is deliberately a pure tensor module — it knows
nothing about knowledge graphs, vocabularies or anonymization. Turning a
human-level query like ``("Alice", "olderThan", "Bob")`` into the tensors it
expects requires a KG (to extract the subgraph), a vocab (to tokenize) and an
anonymization pass. `Scorer` closes over exactly those, so callers get a clean

    scorer.score(anchor, relation, candidate)   # one logit
    scorer.rank(anchor, relation)               # all tails, ranked

without touching tensors. It is the single entry point used by `predict.py`,
the inductive family demo (`tests/test_inductive_family.py`), and anywhere else
that needs "score this triple".

Efficiency: every method routes through `eval.score_candidates`, which encodes
the anchor's subgraph **once** and then varies only the candidate token — so
ranking N tails costs one subgraph encode + one batched classifier pass, not N
full forwards. Build the `Scorer` once and reuse it across many queries; the
context `KnowledgeGraph` (the expensive-to-build part) is constructed a single
time and held.

Supports both scoring modes transparently. Single-tower (default): the candidate
is a token in the anchor's subgraph. Dual-tower (`cfg["dual_subgraph"]`): each
candidate is represented by its OWN encoded k-hop subgraph via a precomputed
per-entity table, built once at construction (mirrors `eval.build_candidate_table`)
so scoring stays fast across many queries.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import torch

from .dataset import KnowledgeGraph, Triple, augment_with_inverse, read_triples
from .eval import filter_known_relations
from .model import InductiveKGModel, load_bundle
from .scoring import build_candidate_table, score_candidates, score_candidates_dual


@dataclass
class Scorer:
    """Scores (anchor, relation, candidate) queries against a fixed context KG."""

    model: InductiveKGModel
    kg: KnowledgeGraph
    vocab: dict[str, int]
    fixed_values: set[str]
    cfg: dict
    device: torch.device

    # Dual-tower candidate table (entity → row index, [|E|, d] vectors). Stays
    # None in single-tower mode; populated by __post_init__ when dual_subgraph.
    _cand_index: dict[str, int] | None = field(default=None, init=False, repr=False)
    _cand_table: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Dual-subgraph checkpoints represent each candidate by its OWN encoded
        # k-hop subgraph rather than as a token in the anchor's neighborhood.
        # The candidate tower depends only on the entity's subgraph (not the
        # anchor/relation), so precompute the per-entity table ONCE here and
        # reuse it across every query — mirrors eval.build_candidate_table.
        if not self.cfg.get("dual_subgraph"):
            return
        if getattr(self.model, "z_mode", "learned") == "runtime":
            # The table and each anchor encode draw independent random [Z_*]
            # banks, so a single score is noisy (eval marginalizes over eval_mc
            # draws; this one-shot facade can't). Stable only for learned mode.
            warnings.warn(
                "Scorer with dual_subgraph + z_mode='runtime': candidate-table "
                "and anchor encodings use independent random [Z_*] draws, so "
                "individual scores are noisy. Reliable only for z_mode='learned'."
            )
        self._cand_index, self._cand_table = build_candidate_table(
            self.model, self.kg, list(self.kg.entities), self.vocab,
            self.fixed_values, self.cfg["max_triples"], self.cfg["z_pool_size"],
            self.device,
            collapse_z=self.cfg.get("collapse_z", False),
            subgraph_hops=self.cfg.get("subgraph_hops", 2),
            use_hop_distance_tokens=self.cfg.get("use_hop_distance_tokens", False),
        )

    # --- construction ------------------------------------------------------

    @classmethod
    def from_bundle(
        cls,
        model_path: str | Path,
        context: str | Path | Iterable[Triple] | KnowledgeGraph,
    ) -> "Scorer":
        """Load a saved bundle and attach a context graph for subgraph extraction.

        `context` may be a triples-file path, an iterable of triples, or a
        prebuilt `KnowledgeGraph`.
        """
        model, vocab, fixed_values, cfg, device = load_bundle(model_path)
        kg = cls._build_kg(context, vocab, cfg)
        return cls(model, kg, vocab, fixed_values, cfg, device)

    @classmethod
    def from_components(
        cls,
        model: InductiveKGModel,
        vocab: dict[str, int],
        fixed_values: set[str],
        cfg: dict,
        device: torch.device,
        context: str | Path | Iterable[Triple] | KnowledgeGraph,
    ) -> "Scorer":
        """Wrap an already-loaded model (e.g. right after training)."""
        kg = cls._build_kg(context, vocab, cfg)
        return cls(model, kg, vocab, fixed_values, cfg, device)

    @staticmethod
    def _build_kg(
        context: str | Path | Iterable[Triple] | KnowledgeGraph,
        vocab: dict[str, int],
        cfg: dict,
    ) -> KnowledgeGraph:
        if isinstance(context, KnowledgeGraph):
            return context
        if isinstance(context, (str, Path)):
            triples = read_triples(context, fmt=cfg.get("triple_format", "head_relation_tail"))
        else:
            triples = list(context)
        # Drop triples whose relation the model never saw, then make the graph
        # bidirectional so inverse relations appear in extracted subgraphs.
        triples, _ = filter_known_relations(triples, vocab)
        return KnowledgeGraph(augment_with_inverse(triples))

    # --- scoring -----------------------------------------------------------

    def score_many(
        self,
        anchor: str,
        relation: str,
        candidates: Sequence[str],
        *,
        exclude_triple: Triple | None = None,
    ) -> dict[str, float]:
        """Score many candidates under one (anchor, relation) → {candidate: logit}.

        Encodes the anchor subgraph once; candidates then vary only the target —
        a token (single-tower) or a precomputed subgraph vector (dual-tower).
        """
        cands = list(candidates)
        if self.cfg.get("dual_subgraph"):
            return score_candidates_dual(
                self.model, anchor, relation, cands,
                self.kg, self.vocab, self.fixed_values,
                self.cfg["max_triples"], self.cfg["z_pool_size"], self.device,
                cand_index=self._cand_index, cand_table=self._cand_table,
                true_cand=None, exclude_triple=exclude_triple,
                collapse_z=self.cfg.get("collapse_z", False),
                subgraph_hops=self.cfg.get("subgraph_hops", 2),
                use_hop_distance_tokens=self.cfg.get("use_hop_distance_tokens", False),
            )
        return score_candidates(
            self.model, anchor, relation, cands,
            self.kg, self.vocab, self.fixed_values,
            max_triples=self.cfg["max_triples"], z_pool=self.cfg["z_pool_size"],
            batch_size=0, device=self.device, exclude_triple=exclude_triple,
            collapse_z=self.cfg.get("collapse_z", False),
            subgraph_hops=self.cfg.get("subgraph_hops", 2),
            use_hop_distance_tokens=self.cfg.get("use_hop_distance_tokens", False),
        )

    def score(
        self,
        anchor: str,
        relation: str,
        candidate: str,
        *,
        exclude_triple: Triple | None = None,
    ) -> float:
        """Score a single (anchor, relation, candidate) triple → logit."""
        return self.score_many(anchor, relation, [candidate], exclude_triple=exclude_triple)[candidate]

    def probability(
        self,
        anchor: str,
        relation: str,
        candidate: str,
        *,
        exclude_triple: Triple | None = None,
    ) -> float:
        """Like `score`, but squashed through a sigmoid → calibrated-ish [0, 1]."""
        return torch.sigmoid(torch.tensor(self.score(anchor, relation, candidate,
                                                     exclude_triple=exclude_triple))).item()

    def rank(
        self,
        anchor: str,
        relation: str,
        *,
        candidates: Sequence[str] | None = None,
        k: int | None = None,
        exclude_triple: Triple | None = None,
    ) -> list[tuple[str, float]]:
        """Rank candidates by score, descending → [(candidate, logit), ...].

        Defaults to ranking every entity in the context KG; pass `candidates`
        to restrict the pool and `k` to keep only the top-k.
        """
        pool = list(candidates) if candidates is not None else list(self.kg.entities)
        scores = self.score_many(anchor, relation, pool, exclude_triple=exclude_triple)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked[:k] if k is not None else ranked
