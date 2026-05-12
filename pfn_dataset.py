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
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

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
    List of ``(triples, entity_embs, relation_embs, entity_to_triples, entity_to_neighbors)``
    tuples — one per KG.  ``triples`` is a list of ``(h_idx, r_idx, t_idx)`` integer
    tuples that index into ``entity_embs`` / ``relation_embs``.
    ``entity_embs`` and ``relation_embs`` are FloatTensors of shape
    ``(n_ent, 384)`` and ``(n_rel, 384)`` produced by all-MiniLM-L6-v2.
    ``entity_to_triples`` maps each entity index to the list of triple indices
    in which that entity appears (as head or tail).
    ``entity_to_neighbors`` stores the undirected entity graph adjacency list
    used for hop-wise context expansion.
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
        if len(raw) > max_per_kg:                  # large KGs (YAGO, FB15k) would otherwise
            raw = random.sample(raw, max_per_kg)   # dominate the task distribution; downsample

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
        pools.append((triples, entity_embs, relation_embs, entity_to_triples, entity_to_neighbors))

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
        self.kg_pools = kg_pools  # list of (triples, entity_embs, relation_embs, entity_to_triples, entity_to_neighbors) per KG

    def _generate_real_task(
        self,
        context_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one binary-labelled episode from a real KG.

        Entity-centric sampling strategy:

        a. Select a random KG.
        b. Select a random focal entity ``e`` and collect its 1-hop neighbourhood
           — all triples in the KG where ``e`` appears as head or tail.
        c. From the neighbourhood, sample the **query** triple (positive example).
          d. Build support by expanding from the focal entity in hop order
              (1-hop, then 2-hop, ... up to ``max_hop``), adding unique triples
              without repetition.
        e. With probability 0.5 keep the query as-is (label=1); otherwise replace
           the tail with a random entity from the full KG (label=0).
        f. Look up pre-computed SentenceTransformer embeddings to build tensors.

        Returns
        -------
        support      : FloatTensor (context_size, 3, ST_DIM)
        query_triple : FloatTensor (3, ST_DIM)  — [head_emb, rel_emb, tail_emb]
        label        : FloatTensor scalar  — 1.0 = real, 0.0 = corrupted
        """

        # TODO:CD: We only corrupt the tail entity for negative examples.  
        # As context_size increases, 
        # We firstly add one-hop triples of the focal entity, then two-hop triples, and so on.
        # This ensures that the support set grows in a structured manner, maintaining local relevance.

        
        # a. Pick one KG; unpack triples + pre-computed ST embedding matrices.
        triples, entity_embs, relation_embs, entity_to_triples, entity_to_neighbors = random.choice(self.kg_pools)
        n_ent = entity_embs.shape[0]

        # b. Pick a focal entity and get its 1-hop neighbourhood triple indices.
        focal = random.randrange(n_ent)
        neighbourhood = entity_to_triples.get(focal, [])
        # Fall back to a random triple when the entity has no known triples.
        if len(neighbourhood) < 2:
            neighbourhood = list(range(len(triples)))

        # c. Sample the query triple from the neighbourhood.
        q_tri_idx = random.choice(neighbourhood)
        q_h, q_r, q_t = triples[q_tri_idx]

        # d. Build support in hop order without duplicated triples.
        ctx_idxs: List[int] = []
        chosen = {q_tri_idx}
        visited_entities = {focal}
        frontier = {focal}

        for _hop in range(max(1, self.max_hop)):
            if len(ctx_idxs) >= context_size or not frontier:
                break

            hop_candidates: List[int] = []
            next_frontier = set()

            for ent in frontier:
                for tri_idx in entity_to_triples.get(ent, []):
                    if tri_idx in chosen:
                        continue
                    chosen.add(tri_idx)
                    hop_candidates.append(tri_idx)

                for nbr in entity_to_neighbors.get(ent, []):
                    if nbr not in visited_entities:
                        visited_entities.add(nbr)
                        next_frontier.add(nbr)

            random.shuffle(hop_candidates)
            space = context_size - len(ctx_idxs)
            ctx_idxs.extend(hop_candidates[:space])
            frontier = next_frontier

        # If hop-limited expansion is insufficient, fill from remaining triples
        # uniformly without replacement to keep support duplicate-free.
        if len(ctx_idxs) < context_size:
            remaining = [i for i in range(len(triples)) if i not in chosen]
            random.shuffle(remaining)
            space = context_size - len(ctx_idxs)
            ctx_idxs.extend(remaining[:space])

        # If the KG is smaller than context_size, repeat triples (with replacement)
        # to fill the remaining slots.  The transformer tolerates duplicate tokens
        # in the support — it just re-encodes the same triple, which is harmless.
        if len(ctx_idxs) < context_size:
            all_available = [i for i in range(len(triples)) if i != q_tri_idx]
            if not all_available:
                all_available = list(range(len(triples)))
            shortfall = context_size - len(ctx_idxs)
            ctx_idxs.extend(random.choices(all_available, k=shortfall))

        support_raw = [triples[i] for i in ctx_idxs]

        # e. Decide label: 50 % positive, 50 % corrupted negative.
        label = int(random.random() < 0.5)

        # f. Build float support tensor from pre-computed embeddings.
        h_list = [h for h, _r, _t in support_raw]
        r_list = [_r for _h, _r, _t in support_raw]
        t_list = [_t for _h, _r, _t in support_raw]
        sup_h = entity_embs[h_list]    # (S, ST_DIM)
        sup_r = relation_embs[r_list]  # (S, ST_DIM)
        sup_t = entity_embs[t_list]    # (S, ST_DIM)
        support_tensor = torch.stack([sup_h, sup_r, sup_t], dim=1)  # (S, 3, ST_DIM)

        # Build query triple embedding.
        q_h_emb = entity_embs[q_h]    # (ST_DIM,)
        q_r_emb = relation_embs[q_r]  # (ST_DIM,)
        if label == 1:
            q_t_emb = entity_embs[q_t]
        else:
            neg_idx = random.choice([i for i in range(n_ent) if i != q_t])
            q_t_emb = entity_embs[neg_idx]

        query_tensor = torch.stack([q_h_emb, q_r_emb, q_t_emb], dim=0)  # (3, ST_DIM)

        return support_tensor, query_tensor, torch.tensor(float(label))

    def generate_task(
        self,
        context_size: int = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one binary-labelled training episode.

        Delegates to :meth:`_generate_real_task`.  ``kg_pools`` must be set
        (pass ``kg_dir`` to :func:`train`).

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
        return self._generate_real_task(context_size)


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
    for i in range(num_episodes):
        if i % log_every == 0:
            print(f"    {i:>7,} / {num_episodes:,}")
        sup, qry, lbl = prior.generate_task(context_size=context_size)
        sup_mm[i] = sup.numpy()
        qry_mm[i] = qry.numpy()
        lbl_mm[i] = lbl.item()

    sup_mm.flush()
    qry_mm.flush()
    lbl_mm.flush()
    del sup_mm, qry_mm, lbl_mm

    meta = {"num_episodes": num_episodes, "context_size": context_size, "st_dim": _ST_DIM}
    with open(os.path.join(path, "dataset_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"  Done  ({num_episodes:,} episodes, context_size={context_size})\n")


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
