"""Dataset utilities for GraphPFN: ST encoder, KG data prior, episode dataset.

Exposes
-------
- ``_ST_DIM`` — output dimensionality of the frozen SentenceTransformer (384).
- ``_encode_strings`` — encode a list of strings to a FloatTensor via all-MiniLM-L6-v2.
- ``_load_real_triples`` — walk a KG directory tree and load indexed triple pools.
- ``RichSubgraphPrior`` — entity-centric episode sampler from real KG pools.
- ``_collate`` — stack a list of (support, query, label) tuples into a batch.
- ``build_dataset`` — pre-generate episodes and persist as memory-mapped .npy files.
- ``PFNDataset`` — memory-mapped ``torch.utils.data.Dataset`` of pre-built episodes.
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset

# ── Sentence-Transformer embedding cache ────────────────────────────────────
_ST_MODEL: Optional[SentenceTransformer] = None
_ST_DIM = 384   # output dimension of all-MiniLM-L6-v2


def _get_st_model() -> SentenceTransformer:
    """Lazy-load and cache sentence-transformers/all-MiniLM-L6-v2."""
    global _ST_MODEL
    if _ST_MODEL is None:
        print("  Loading sentence-transformers/all-MiniLM-L6-v2 ...")
        _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _ST_MODEL.eval()
        for p in _ST_MODEL.parameters():
            p.requires_grad_(False)
    return _ST_MODEL


def _encode_strings(strings: List[str]) -> torch.Tensor:
    """Encode a list of strings → FloatTensor (N, 384) via all-MiniLM-L6-v2."""
    st = _get_st_model()
    embs = st.encode(strings, batch_size=512, show_progress_bar=False, convert_to_numpy=True)
    return torch.from_numpy(np.array(embs)).float()


# ---------------------------------------------------------------------------
# 1.  DATA PRIOR
# ---------------------------------------------------------------------------

def _load_real_triples(
    kg_dir: str,
    max_per_kg: int = 500,
) -> List[
    Tuple[
        List[Tuple[int, int, int]],
        torch.Tensor,
        torch.Tensor,
        Dict[int, List[int]],
        Dict[int, List[int]],
        List[str],
        List[str],
    ]
]:
    """Load all ``train.txt`` files under *kg_dir* as episodic task pools.

    Each KG is kept separate so that tasks sample context triples from a single
    graph (preserving relational structure).  Entity and relation tokens are
    encoded to 384-dimensional vectors by the frozen all-MiniLM-L6-v2
    SentenceTransformer and stored as float tensors.  No ID re-randomisation
    is performed — semantic identity is captured by the embedding itself.

    Parameters
    ----------
    kg_dir : str
        Root directory to walk recursively for ``train.txt`` files.
    max_per_kg : int
        Maximum number of triples to retain per KG.  Large KGs are randomly
        downsampled so they do not dominate the task distribution.

    Returns
    -------
    List of ``(triples, entity_embs, relation_embs, entity_to_triples, entity_to_neighbors,
    entity_strings, relation_strings)`` tuples — one per KG.  ``triples`` is a list of
    ``(h_idx, r_idx, t_idx)`` integer tuples that index into ``entity_embs`` / ``relation_embs``.
    ``entity_embs`` and ``relation_embs`` are FloatTensors of shape ``(n_ent, 384)`` and
    ``(n_rel, 384)`` produced by all-MiniLM-L6-v2.  ``entity_to_triples`` maps each entity
    index to the list of triple indices in which that entity appears (as head or tail).
    ``entity_to_neighbors`` stores the undirected entity graph adjacency list used for
    hop-wise context expansion.  ``entity_strings`` and ``relation_strings`` are lists
    of the original token strings, indexed consistently with entity/relation embeddings.
    """
    pools = []
    for root, _dirs, files in os.walk(kg_dir):     # walk the directory tree recursively
        if "train.txt" not in files:               # skip folders that have no training file
            continue
        path = os.path.join(root, "train.txt")     # full path to this KG's training file
        raw: List[Tuple[str, str, str]] = []       # will hold raw string triples
        with open(path) as fh:
            for line in fh:
                parts = line.strip().split()       # split on any whitespace (tab or space)
                if len(parts) == 3:                # skip blank lines / malformed rows
                    raw.append((parts[0], parts[1], parts[2]))
        if not raw:                                # entirely empty file — nothing to learn from
            continue
        #if len(raw) > max_per_kg:                  # large KGs (YAGO, FB15k) would otherwise
        #    raw = random.sample(raw, max_per_kg)   # dominate the task distribution; downsample

        entity_vocab: Dict[str, int] = {}          # string entity → local integer index
        relation_vocab: Dict[str, int] = {}        # string relation → local integer index
        triples: List[Tuple[int, int, int]] = []   # indexed (h, r, t) triples for this KG
        for h, r, t in raw:
            if h not in entity_vocab:              # assign new index the first time we see this entity
                entity_vocab[h] = len(entity_vocab)
            if t not in entity_vocab:              # same for the tail entity
                entity_vocab[t] = len(entity_vocab)
            if r not in relation_vocab:            # same for the relation
                relation_vocab[r] = len(relation_vocab)
            triples.append((entity_vocab[h], relation_vocab[r], entity_vocab[t]))

        kg_name = os.path.basename(root)           # short name for logging (folder name)

        # Encode entity and relation token strings to 384-dim ST embeddings.
        entity_strings   = [tok for tok, _ in sorted(entity_vocab.items(), key=lambda x: x[1])]
        relation_strings = [tok for tok, _ in sorted(relation_vocab.items(), key=lambda x: x[1])]
        entity_embs   = _encode_strings(entity_strings)    # (n_ent, ST_DIM) – CPU
        relation_embs = _encode_strings(relation_strings)  # (n_rel, ST_DIM) – CPU

        # Build entity → triple-index mapping and entity adjacency for hop expansion.
        entity_to_triples: Dict[int, List[int]] = {}
        entity_to_neighbors_set: Dict[int, set] = {}
        for tri_idx, (h, _r, t) in enumerate(triples):
            entity_to_triples.setdefault(h, []).append(tri_idx)
            entity_to_triples.setdefault(t, []).append(tri_idx)
            entity_to_neighbors_set.setdefault(h, set()).add(t)
            entity_to_neighbors_set.setdefault(t, set()).add(h)

        entity_to_neighbors: Dict[int, List[int]] = {
            ent: list(neis) for ent, neis in entity_to_neighbors_set.items()
        }

        print(
            f"  {kg_name:<30s}  {len(triples):>6d} triples  "
            f"{len(entity_vocab):>5d} entities  {len(relation_vocab):>4d} relations"
        )
        pools.append((triples, entity_embs, relation_embs, entity_to_triples, entity_to_neighbors,
                      entity_strings, relation_strings))

    return pools


class RichSubgraphPrior:
    """Generates binary triple-scoring episodes from real knowledge graphs.

    Each episode consists of:

        - A **support set** of up to ``context_size`` triples built by hop-wise
            expansion around a randomly chosen focal entity (1-hop, then 2-hop,
            ... up to ``max_hop``), represented
      as FloatTensor ``(context_size, 3, 384)`` of SentenceTransformer embeddings.
    - A **query triple** ``(h, r, t)`` also drawn from the same neighbourhood,
      represented as FloatTensor ``(3, 384)``.
    - A **label** — ``1.0`` if the query is a real KG triple (positive),
      ``0.0`` if its tail has been replaced with a random entity (negative).

    The entity-centric sampling strategy keeps support context locally relevant
    while allowing broader structural coverage as context grows.  Triples are
    added without repetition, prioritising closer hops first.

    Entity and relation identity is captured by frozen SentenceTransformer
    embeddings, so **no ID re-randomisation** is needed during training.
    """

    def __init__(
        self,
        max_hop: int = 3,
        kg_pools: Optional[list] = None,
    ):
        self.max_hop = max_hop
        # list of (triples, entity_embs, relation_embs, entity_to_triples, entity_to_neighbors,
        # entity_strings, relation_strings) per KG
        self.kg_pools = kg_pools

    def _generate_real_task(
        self,
        context_size: int,
        force_label: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one binary-labelled episode from a real KG via entity-centric hop-wise sampling.

        The episode construction strategy ensures that the support context is both
        **locally coherent** (grounded in a focal entity's neighborhood) and
        **structurally diverse** (expanding across multiple hops to reach distant
        entities while maintaining connectivity).  Neighbors are handled
        intelligently via a frontier-based BFS: if an entity appears in triples
        from multiple parents in the current hop, it is discovered only once
        (tracked by ``visited_entities`` set) and expanded once (added to
        ``next_frontier`` set), avoiding redundant processing while maximizing
        unique neighbor discovery.

        **Algorithm outline:**

        1. **Select a random KG** from the loaded pools.  Each KG is treated as
           a separate episodic task source to preserve relational structure.

        2. **Pick a random focal entity** ``e`` and collect its 1-hop neighbourhood
           — the set of all triples where ``e`` appears as head or tail.  If the
           focal entity has very few or no triples, fall back to all triples
           in the KG.

        3. **Sample the query triple** uniformly from the focal entity's
           neighbourhood.  This ensures the query is semantically relevant to
           the support context.

        4. **Expand the support set in hop order** using a frontier-based BFS:
           - **Hop 0 (initial):** Start with the focal entity as the frontier.
           - **Each hop:** Collect all unvisited triples of frontier entities
             (avoiding re-selection via ``chosen`` set) and discover their
             neighboring entities (avoiding re-visitation via ``visited_entities``
             set).  Shuffle candidate triples and add up to ``space`` of them
             to the support set.
           - **Frontier advancement:** After processing all frontier entities,
             move all newly discovered entities to the next frontier.  A neighbor
             might appear in multiple triples from different frontier entities;
             the ``visited_entities`` set ensures it is only expanded once, while
             the triples connecting to it are individually selected for inclusion
             in support.
           - **Termination:** Stop expanding when context_size is reached or
             no new neighbors can be discovered.

        5. **Fill remaining slots** (if any):
           - First, try to sample from the remaining unexplored triples
             (no repetition) to preserve diversity.
           - If the KG is smaller than context_size, pad with repetitions
             from available triples.  The transformer handles duplicate tokens
             gracefully by re-encoding the same triple, which is harmless.

        6. **Decide label:** Either keep the query as-is (label=1, positive),
           or replace its tail with a random entity from the full KG (label=0,
           negative).  The ``force_label`` parameter allows external control
           for ratio scheduling during dataset generation.

        7. **Build tensor representations** by looking up pre-computed
           SentenceTransformer embeddings (384-dim) for all entities and
           relations in the support and query.

        Parameters
        ----------
        verbose : bool
            If True, print the string representations of support triples, query,
            and label for debugging purposes.

        Returns
        -------
        support      : FloatTensor (context_size, 3, ST_DIM)
            Context triple embeddings, padded with duplicates if needed.
        query_triple : FloatTensor (3, ST_DIM)
            [head_emb, relation_emb, tail_emb] — the candidate triple.
        label        : FloatTensor scalar
            1.0 if query is a real KG triple, 0.0 if tail is negatively sampled.
        """

        # TODO:CD: Currently we only corrupt the tail entity for negative examples.
        # Consider extending to head corruption or relation corruption for richer
        # learning signals.  As context_size increases, we incrementally add
        # one-hop triples, then two-hop triples, and so on, ensuring the support
        # set grows in a structured manner while maintaining local relevance.

        # ── STEP 1: SELECT A RANDOM KG ──────────────────────────────────────
        # Pick one KG from the pool; unpack its triples, embeddings, and
        # vocabulary strings for potential debug output.
        triples, entity_embs, relation_embs, entity_to_triples, entity_to_neighbors, entity_strings, relation_strings = random.choice(self.kg_pools)
        n_ent = entity_embs.shape[0]

        # ── STEP 2: PICK A RANDOM FOCAL ENTITY & GET ITS NEIGHBORHOOD ──────
        # The focal entity anchors the entire episode; all support and query
        # triples will be sampled from its neighborhood or further via BFS.
        focal = random.randrange(n_ent)
        neighbourhood = entity_to_triples.get(focal, [])
        # Fallback: if the focal entity has no triples (isolated), sample from
        # all KG triples.  This prevents the episode generator from crashing
        # on sparse entities.
        if len(neighbourhood) < 2:
            neighbourhood = list(range(len(triples)))

        # ── STEP 3: SAMPLE THE QUERY TRIPLE FROM THE NEIGHBORHOOD ──────────
        # This ensures the query is semantically close to the focal entity,
        # making the learning task more coherent.
        q_tri_idx = random.choice(neighbourhood)
        q_h, q_r, q_t = triples[q_tri_idx]

        # ── STEP 4: BUILD SUPPORT VIA HOP-WISE BFS EXPANSION ────────────────
        # Initialize the frontier-based graph traversal.
        #   - ctx_idxs: list of triple indices selected for the support set.
        #   - chosen: set of triple indices already added (prevents duplicates).
        #   - visited_entities: set of entities already visited (prevents re-expansion).
        #   - frontier: current set of entities to expand from.
        ctx_idxs: List[int] = []
        chosen = {q_tri_idx}
        visited_entities = {focal}
        frontier = {focal}

        # BFS expansion: iterate through each hop up to max_hop.
        for _hop in range(max(1, self.max_hop)):
            # Early termination conditions:
            #   1. Support size already meets context_size.
            #   2. No new entities to expand (frontier exhausted).
            if len(ctx_idxs) >= context_size or not frontier:
                break

            hop_candidates: List[int] = []  # triples reachable from current frontier
            next_frontier = set()           # entities to expand in the next hop

            # ─ Expand from each entity in the current frontier ─
            for ent in frontier:
                # Collect all triples involving this entity that haven't
                # been selected yet.  A single entity might appear in multiple
                # triples; we add all of them to the candidate pool, then
                # intelligently shuffle and sample to maintain diversity.
                for tri_idx in entity_to_triples.get(ent, []):
                    if tri_idx in chosen:
                        continue  # Skip already-selected triples.
                    chosen.add(tri_idx)
                    hop_candidates.append(tri_idx)

                # Discover neighbors (unvisited entities connected to ent).
                # A neighbor might appear in multiple triples from different
                # frontier entities; the visited_entities set ensures it is
                # only added to next_frontier once, avoiding redundant re-visitation.
                for nbr in entity_to_neighbors.get(ent, []):
                    if nbr not in visited_entities:
                        visited_entities.add(nbr)
                        next_frontier.add(nbr)

            # Add candidates to support set: shuffle to randomize hop-local
            # order, then take up to 'space' candidates to fill the current
            # support budget.  This balances hop-wise expansion (closer
            # entities first) with random sampling within each hop.
            random.shuffle(hop_candidates)
            space = context_size - len(ctx_idxs)
            ctx_idxs.extend(hop_candidates[:space])
            frontier = next_frontier

        # ── STEP 4b: FILL FROM REMAINING UNEXPLORED TRIPLES ────────────────
        # If hop expansion did not reach context_size, try to fill the support
        # by sampling from triples we haven't visited yet.  This maximizes
        # knowledge diversity without repetition.
        if len(ctx_idxs) < context_size:
            remaining = [i for i in range(len(triples)) if i not in chosen]
            random.shuffle(remaining)
            space = context_size - len(ctx_idxs)
            ctx_idxs.extend(remaining[:space])

        # ── STEP 4c: PAD WITH DUPLICATES IF NECESSARY ──────────────────────
        # Some KGs (especially small ones like Countries-S1 with ~150 triples)
        # may be smaller than context_size.  We pad with repetitions of
        # available triples.  The transformer handles duplicate tokens
        # gracefully: it re-encodes the same triple multiple times, which
        # provides redundant information but does not break the forward pass.
        if len(ctx_idxs) < context_size:
            all_available = [i for i in range(len(triples)) if i != q_tri_idx]
            if not all_available:
                all_available = list(range(len(triples)))
            shortfall = context_size - len(ctx_idxs)
            ctx_idxs.extend(random.choices(all_available, k=shortfall))

        support_raw = [triples[i] for i in ctx_idxs]

        # ── STEP 5: DECIDE LABEL (POSITIVE OR NEGATIVE) ──────────────────────
        # Determine whether the query is a real triple (label=1) or a negatively
        # sampled triple with a corrupted tail (label=0).  The force_label
        # parameter allows external scheduling (e.g., for 1:N pos:neg ratio).
        if force_label is None:
            label = int(random.random() < 0.5)
        else:
            label = int(force_label)

        # ── STEP 6: BUILD FLOAT TENSORS FROM PRE-COMPUTED EMBEDDINGS ────────
        # Unpack entity and relation indices from support triples and look up
        # their pre-computed SentenceTransformer embeddings (384-dim).
        h_list = [h for h, _r, _t in support_raw]
        r_list = [_r for _h, _r, _t in support_raw]
        t_list = [_t for _h, _r, _t in support_raw]
        sup_h = entity_embs[h_list]    # (S, ST_DIM)
        sup_r = relation_embs[r_list]  # (S, ST_DIM)
        sup_t = entity_embs[t_list]    # (S, ST_DIM)
        support_tensor = torch.stack([sup_h, sup_r, sup_t], dim=1)  # (S, 3, ST_DIM)

        # Build query triple embedding: [head_emb, rel_emb, tail_emb].
        # If label=1, the tail is real; if label=0, it's a random entity.
        q_h_emb = entity_embs[q_h]    # (ST_DIM,)
        q_r_emb = relation_embs[q_r]  # (ST_DIM,)
        if label == 1:
            q_t_actual = q_t
            q_t_emb = entity_embs[q_t]
        else:
            # Negative sampling: replace tail with a random entity different
            # from the true tail.
            q_t_actual = random.choice([i for i in range(n_ent) if i != q_t])
            q_t_emb = entity_embs[q_t_actual]

        query_tensor = torch.stack([q_h_emb, q_r_emb, q_t_emb], dim=0)  # (3, ST_DIM)

        # ── OPTIONAL: DEBUG OUTPUT ──────────────────────────────────────────
        # Print string representations of support, query, and label for debugging.
        if verbose:
            print("\n  ─── Episode Debug Output ───")
            print("  Support triples (string):")
            for tri_idx in ctx_idxs:
                h, r, t = triples[tri_idx]
                print(f"    ({entity_strings[h]}, {relation_strings[r]}, {entity_strings[t]})")
            q_h_str = entity_strings[q_h]
            q_r_str = relation_strings[q_r]
            q_t_actual_str = entity_strings[q_t_actual]
            print(f"  Query triple (string): ({q_h_str}, {q_r_str}, {q_t_actual_str})")
            if label == 0:
                q_t_true_str = entity_strings[q_t]
                print(f"    True tail (for reference): {q_t_true_str}")
                print(f"    Sampled tail (used): {q_t_actual_str}")
            print(f"  Label: {label} ({'real' if label == 1 else 'negatively sampled'})")
            print("  ───────────────────────────")

        return support_tensor, query_tensor, torch.tensor(float(label))

    def generate_task(
        self,
        context_size: int = 1000,
        force_label: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one binary-labelled training episode.

        Delegates to :meth:`_generate_real_task`.  ``kg_pools`` must be set
        (pass ``kg_dir`` to :func:`train`).

        Parameters
        ----------
        verbose : bool
            If True, print string representations of support, query, and label.

        Returns
        -------
        support      : FloatTensor (context_size, 3, ST_DIM)  — context triples
        query_triple : FloatTensor (3, ST_DIM)                — candidate triple
        label        : FloatTensor scalar                     — 1.0 real, 0.0 corrupted
        """
        if not self.kg_pools:
            raise ValueError(
                "kg_pools is empty.  Pass --kg-dir (CLI) or kg_dir (API) to load real KGs."
            )
        return self._generate_real_task(context_size, force_label=force_label, verbose=verbose)


class RandomSupportPrior:
    """Generates binary triple-scoring episodes with random support sampling.

    Unlike :class:`RichSubgraphPrior` (entity-centric neighborhood expansion),
    this prior samples support triples uniformly from the KG pool, which can be
    useful when support sets in deployment are not tightly centered around a
    focal entity.

    A random query triple is selected from the same KG.  Labels are generated
    exactly like the entity-centric prior:

    - ``label=1``: query remains a true triple.
    - ``label=0``: query tail is replaced with a random entity.

    Optionally, support order can be permuted per episode so the model sees
    many orderings of the same structural evidence.
    """

    def __init__(
        self,
        kg_pools: Optional[list] = None,
        permute_support: bool = False,
    ):
        # list of (triples, entity_embs, relation_embs, entity_to_triples, entity_to_neighbors,
        # entity_strings, relation_strings) per KG
        self.kg_pools = kg_pools
        self.permute_support = permute_support

    def _generate_real_task(
        self,
        context_size: int,
        force_label: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one binary-labelled episode with random support triples."""
        triples, entity_embs, relation_embs, _e2t, _e2n, entity_strings, relation_strings = random.choice(self.kg_pools)
        n_ent = entity_embs.shape[0]

        q_tri_idx = random.randrange(len(triples))
        q_h, q_r, q_t = triples[q_tri_idx]

        all_idxs = [i for i in range(len(triples)) if i != q_tri_idx]
        if len(all_idxs) >= context_size:
            ctx_idxs = random.sample(all_idxs, k=context_size)
        else:
            ctx_idxs = all_idxs[:]
            shortfall = context_size - len(ctx_idxs)
            if not ctx_idxs:
                ctx_idxs = [q_tri_idx] * context_size
            else:
                ctx_idxs.extend(random.choices(ctx_idxs, k=shortfall))

        if self.permute_support:
            random.shuffle(ctx_idxs)

        support_raw = [triples[i] for i in ctx_idxs]

        if force_label is None:
            label = int(random.random() < 0.5)
        else:
            label = int(force_label)

        h_list = [h for h, _r, _t in support_raw]
        r_list = [_r for _h, _r, _t in support_raw]
        t_list = [_t for _h, _r, _t in support_raw]
        sup_h = entity_embs[h_list]
        sup_r = relation_embs[r_list]
        sup_t = entity_embs[t_list]
        support_tensor = torch.stack([sup_h, sup_r, sup_t], dim=1)

        q_h_emb = entity_embs[q_h]
        q_r_emb = relation_embs[q_r]
        if label == 1:
            q_t_actual = q_t
            q_t_emb = entity_embs[q_t]
        else:
            q_t_actual = random.choice([i for i in range(n_ent) if i != q_t])
            q_t_emb = entity_embs[q_t_actual]

        query_tensor = torch.stack([q_h_emb, q_r_emb, q_t_emb], dim=0)

        if verbose:
            print("\n  ─── Episode Debug Output (RandomSupportPrior) ───")
            print("  Support triples (string):")
            for tri_idx in ctx_idxs:
                h, r, t = triples[tri_idx]
                print(f"    ({entity_strings[h]}, {relation_strings[r]}, {entity_strings[t]})")
            q_h_str = entity_strings[q_h]
            q_r_str = relation_strings[q_r]
            q_t_actual_str = entity_strings[q_t_actual]
            print(f"  Query triple (string): ({q_h_str}, {q_r_str}, {q_t_actual_str})")
            if label == 0:
                q_t_true_str = entity_strings[q_t]
                print(f"    True tail (for reference): {q_t_true_str}")
                print(f"    Sampled tail (used): {q_t_actual_str}")
            print(f"  Label: {label} ({'real' if label == 1 else 'negatively sampled'})")
            print("  ─────────────────────────────────────────────────")

        return support_tensor, query_tensor, torch.tensor(float(label))

    def generate_task(
        self,
        context_size: int = 1000,
        force_label: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one binary-labelled training episode."""
        if not self.kg_pools:
            raise ValueError(
                "kg_pools is empty.  Pass --kg-dir (CLI) or kg_dir (API) to load real KGs."
            )
        return self._generate_real_task(context_size, force_label=force_label, verbose=verbose)


def _collate(
    tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack a list of (support, query_triple, label) tuples into batched tensors."""
    return (
        torch.stack([t[0] for t in tasks]),   # (B, S, 3, ST_DIM)
        torch.stack([t[1] for t in tasks]),   # (B, 3, ST_DIM)
        torch.stack([t[2] for t in tasks]),   # (B,)  float labels
    )


# ---------------------------------------------------------------------------
# PRE-COMPUTED EPISODE DATASET
# ---------------------------------------------------------------------------

def build_dataset(
    prior: RichSubgraphPrior,
    num_episodes: int,
    context_size: int,
    path: str,
    negative_ratio: int = 1,
    verbose: bool = False,
) -> None:
    """Pre-generate training episodes and persist them as memory-mapped .npy files.

    Three binary files are written inside *path*:

    - ``dataset_supports.npy``  —  float32, shape ``(N, context_size, 3, ST_DIM)``
    - ``dataset_queries.npy``   —  float32, shape ``(N, 3, ST_DIM)``
    - ``dataset_labels.npy``    —  float32, shape ``(N,)``

    A small ``dataset_meta.json`` sidecar stores ``num_episodes``,
    ``context_size``, and ``st_dim`` so the loader can validate them at
    read time.

    All files use numpy's ``.npy`` format with a standard header, so they can
    be opened as memory-mapped arrays via ``np.load(file, mmap_mode='r')``.
    This means that even very large datasets (tens of GB) need not reside in
    RAM in full: the OS pages in only the slices that are actually accessed.

    Parameters
    ----------
    prior : RichSubgraphPrior
        Initialised prior with at least one KG pool loaded.
    num_episodes : int
        Total number of episodes to generate.
    context_size : int
        Number of support triples per episode.  Must match the value used
        at training time.
    path : str
        Directory in which to write the dataset files (created if absent).
    negative_ratio : int
        Number of negative examples generated per positive example.  For
        example, ``negative_ratio=10`` yields a 1:10 positive:negative schedule.
    verbose : bool
        If True, print progress logs and episode debug output during generation.
    """
    os.makedirs(path, exist_ok=True)
    sup_path = os.path.join(path, "dataset_supports.npy")
    qry_path = os.path.join(path, "dataset_queries.npy")
    lbl_path = os.path.join(path, "dataset_labels.npy")

    sup_mm = np.lib.format.open_memmap(
        sup_path, mode="w+", dtype=np.float32,
        shape=(num_episodes, context_size, 3, _ST_DIM),
    )
    qry_mm = np.lib.format.open_memmap(
        qry_path, mode="w+", dtype=np.float32,
        shape=(num_episodes, 3, _ST_DIM),
    )
    lbl_mm = np.lib.format.open_memmap(
        lbl_path, mode="w+", dtype=np.float32,
        shape=(num_episodes,),
    )

    log_every = max(1, num_episodes // 10)
    print(f"  Generating {num_episodes:,} episodes  →  {path!r}")
    cycle_len = negative_ratio + 1
    for i in range(num_episodes):
        if i % log_every == 0:
            print(f"    {i:>7,} / {num_episodes:,}")
        force_label = 1 if (i % cycle_len == 0) else 0
        sup, qry, lbl = prior.generate_task(context_size=context_size, verbose=verbose, force_label=force_label)
        sup_mm[i] = sup.numpy()
        qry_mm[i] = qry.numpy()
        lbl_mm[i] = lbl.item()

    sup_mm.flush()
    qry_mm.flush()
    lbl_mm.flush()
    del sup_mm, qry_mm, lbl_mm

    meta = {
        "num_episodes": num_episodes,
        "context_size": context_size,
        "st_dim": _ST_DIM,
        "negative_ratio": negative_ratio,
    }
    with open(os.path.join(path, "dataset_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(
        f"  Done  ({num_episodes:,} episodes, context_size={context_size}, "
        f"pos:neg ≈ 1:{negative_ratio})\n"
    )


class PFNDataset(Dataset):
    """Memory-mapped :class:`torch.utils.data.Dataset` of pre-generated episodes.

    Reads the three ``.npy`` files produced by :func:`build_dataset`:

    - ``dataset_supports.npy``  —  float32, (N, context_size, 3, ST_DIM)
    - ``dataset_queries.npy``   —  float32, (N, 3, ST_DIM)
    - ``dataset_labels.npy``    —  float32, (N,)

    Each array is opened with ``mmap_mode='r'`` so only the pages currently
    accessed reside in RAM.  Multi-worker DataLoader usage is safe because
    the mmap file handles are read-only and independent across workers.

    Parameters
    ----------
    path : str
        Directory containing the three ``.npy`` dataset files.
    expected_context_size : int or None
        When given, raises ``ValueError`` if the on-disk context_size does
        not match (prevents silent shape mismatches between dataset and model).
    """

    def __init__(self, path: str, expected_context_size: Optional[int] = None):
        self.supports = np.load(os.path.join(path, "dataset_supports.npy"), mmap_mode="r")
        self.queries  = np.load(os.path.join(path, "dataset_queries.npy"),  mmap_mode="r")
        self.labels   = np.load(os.path.join(path, "dataset_labels.npy"),   mmap_mode="r")

        self.context_size: int = int(self.supports.shape[1])
        if expected_context_size is not None and self.context_size != expected_context_size:
            raise ValueError(
                f"Dataset at '{path}' has context_size={self.context_size}, "
                f"but training requested context_size={expected_context_size}.  "
                f"Delete the dataset folder or pass --context-size {self.context_size}."
            )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.supports[idx].copy()),   # (context_size, 3, ST_DIM)
            torch.from_numpy(self.queries[idx].copy()),    # (3, ST_DIM)
            torch.tensor(float(self.labels[idx]), dtype=torch.float32),
        )
