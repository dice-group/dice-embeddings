"""Evaluate a saved GraphPFN checkpoint on a test file.

Usage
-----
    python pfn_eval.py \
        --model model.pt \
        --train-file KGs/Countries-S1/train.txt \
        --test-file  KGs/Countries-S1/test.txt  \
        --batch-size 100 \
        --context-size 128

For each test triple (h, r, t) — all assumed correct (label=1) — the model
is given an **entity-centric** support context sampled from the 1-hop
neighbourhood of the query head entity h in train.txt.  This matches the
training distribution exactly: training episodes also use entity-centric
support drawn from the focal entity's neighbourhood.

Using a single global random support (as a naive baseline would) causes a
large distribution mismatch and inflated BCE loss, because the model was
never trained on contexts unrelated to the query entity.
"""

import argparse
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pfn_dataset import _encode_strings
from pfn_model import TriplePFN


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _load_triples(path: str) -> List[Tuple[str, str, str]]:
    """Read a whitespace-separated triple file; return list of (h, r, t) strings."""
    triples = []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


def _build_entity_index(
    triples: List[Tuple[str, str, str]],
) -> Dict[str, List[int]]:
    """Map each entity string → list of triple indices it appears in (head or tail)."""
    index: Dict[str, List[int]] = defaultdict(list)
    for i, (h, _r, t) in enumerate(triples):
        index[h].append(i)
        index[t].append(i)
    return index


def _entity_centric_support(
    query_h: str,
    train_triples: List[Tuple[str, str, str]],
    entity_index: Dict[str, List[int]],
    context_size: int,
    rng: random.Random,
) -> List[Tuple[str, str, str]]:
    """Sample up to *context_size* triples from the neighbourhood of *query_h*.

    Mirrors the entity-centric BFS used in :class:`RichSubgraphPrior`:
    1. Start with the 1-hop triples of *query_h*.
    2. Expand to neighbours' triples if more slots remain.
    3. Pad with random triples if the neighbourhood is smaller than context_size.
    """
    chosen_idxs: set = set()
    support_idxs: List[int] = []

    # 1-hop: all triples containing query_h
    hop1 = list(entity_index.get(query_h, []))
    rng.shuffle(hop1)
    for idx in hop1:
        if len(support_idxs) >= context_size:
            break
        if idx not in chosen_idxs:
            chosen_idxs.add(idx)
            support_idxs.append(idx)

    # 2-hop: triples of neighbours reached in hop-1
    if len(support_idxs) < context_size:
        neighbors: set = set()
        for idx in support_idxs:
            h, _r, t = train_triples[idx]
            neighbors.add(h)
            neighbors.add(t)
        neighbors.discard(query_h)

        hop2: List[int] = []
        for nb in neighbors:
            hop2.extend(entity_index.get(nb, []))
        rng.shuffle(hop2)
        for idx in hop2:
            if len(support_idxs) >= context_size:
                break
            if idx not in chosen_idxs:
                chosen_idxs.add(idx)
                support_idxs.append(idx)

    # Pad with random triples if neighbourhood is smaller than context_size
    if len(support_idxs) < context_size:
        remaining = [i for i in range(len(train_triples)) if i not in chosen_idxs]
        rng.shuffle(remaining)
        for idx in remaining:
            if len(support_idxs) >= context_size:
                break
            support_idxs.append(idx)

    # If the KG is still smaller, repeat (pad with duplicates like the prior does)
    while len(support_idxs) < context_size:
        support_idxs.append(rng.choice(support_idxs))

    return [train_triples[i] for i in support_idxs[:context_size]]


# ---------------------------------------------------------------------------
# MAIN EVALUATION FUNCTION
# ---------------------------------------------------------------------------

def evaluate_bce(
    model: TriplePFN,
    train_file: str,
    test_file: str,
    batch_size: int = 100,
    context_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> float:
    """Compute BCE loss over all test triples (all labels = 1).

    For each test triple (h, r, t) an **entity-centric** support context is
    built from the neighbourhood of h in train.txt, matching the training
    distribution.  Using a global random support would cause a large
    distribution mismatch and inflate the reported loss.

    Parameters
    ----------
    model : TriplePFN
        Trained model.
    train_file : str
        Path to train.txt used as in-context support.
    test_file : str
        Path to test.txt; every triple is treated as a positive (label=1).
    batch_size : int
        Number of test triples per forward pass.
    context_size : int or None
        Number of support triples per query.  Should match the value used
        during training.  None defaults to 32 to avoid sequence-length mismatch.
    device : torch.device or None
        Defaults to CUDA if available, else CPU.
    seed : int
        Random seed used for support sampling.

    Returns
    -------
    float
        Average BCE loss over all (in-vocab) test triples.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    # ── 1. Build shared vocab from train.txt ─────────────────────────────────
    print(f"Loading train triples from '{train_file}' ...")
    train_triples_raw = _load_triples(train_file)
    entity_vocab: Dict[str, int] = {}
    relation_vocab: Dict[str, int] = {}
    for h_s, r_s, t_s in train_triples_raw:
        for tok in (h_s, t_s):
            if tok not in entity_vocab:
                entity_vocab[tok] = len(entity_vocab)
        if r_s not in relation_vocab:
            relation_vocab[r_s] = len(relation_vocab)

    n_ent = len(entity_vocab)
    n_rel = len(relation_vocab)
    print(f"  {len(train_triples_raw):,} train triples | {n_ent} entities | {n_rel} relations")

    # Build entity → triple-index lookup for entity-centric support sampling.
    entity_index = _build_entity_index(train_triples_raw)

    # ── 2. Encode all entity / relation strings ───────────────────────────────
    entity_strings   = [tok for tok, _ in sorted(entity_vocab.items(),   key=lambda x: x[1])]
    relation_strings = [tok for tok, _ in sorted(relation_vocab.items(), key=lambda x: x[1])]
    print("  Encoding strings with SentenceTransformer ...")
    entity_embs   = _encode_strings(entity_strings).to(device)    # (n_ent, 384)
    relation_embs = _encode_strings(relation_strings).to(device)  # (n_rel, 384)

    eff_ctx = context_size if context_size is not None else 32
    if context_size is None:
        print(f"  No --context-size given; defaulting to {eff_ctx} (set explicitly to match training).")
    # ── 4. Load test triples ─────────────────────────────────────────────────
    print(f"Loading test triples from '{test_file}' ...")
    test_triples_raw = _load_triples(test_file)
    n_skipped = 0
    test_triples_in_vocab: List[Tuple[str, str, str]] = []
    for h_s, r_s, t_s in test_triples_raw:
        if h_s not in entity_vocab or t_s not in entity_vocab or r_s not in relation_vocab:
            n_skipped += 1
            continue
        test_triples_in_vocab.append((h_s, r_s, t_s))

    if n_skipped:
        print(f"  Skipped {n_skipped} test triples with OOV tokens.")
    n_test = len(test_triples_in_vocab)
    print(
        f"  Evaluating {n_test:,} test triples (all labels = 1) | "
        f"entity-centric support (context_size={eff_ctx}) ..."
    )

    # ── 5. Iterate over test triples; build entity-centric support per batch ─
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for start in tqdm(
            range(0, n_test, batch_size),
            desc="Evaluating",
            unit="batch",
        ):
            batch = test_triples_in_vocab[start : start + batch_size]
            B = len(batch)

            # Build query tensor: (B, 3, 384)
            q_h_embs = entity_embs[[entity_vocab[h] for h, _r, _t in batch]]
            q_r_embs = relation_embs[[relation_vocab[r] for _h, r, _t in batch]]
            q_t_embs = entity_embs[[entity_vocab[t] for _h, _r, t in batch]]
            query_tensor = torch.stack([q_h_embs, q_r_embs, q_t_embs], dim=1)  # (B, 3, 384)

            # Build entity-centric support for every item in the batch: (B, S, 3, 384)
            sup_list = []
            for h_s, _r_s, _t_s in batch:
                sup_triples = _entity_centric_support(
                    h_s, train_triples_raw, entity_index, eff_ctx, rng
                )
                sup_h = entity_embs[[entity_vocab[h] for h, _r, _t in sup_triples]]
                sup_r = relation_embs[[relation_vocab[r] for _h, r, _t in sup_triples]]
                sup_t = entity_embs[[entity_vocab[t] for _h, _r, t in sup_triples]]
                sup_list.append(torch.stack([sup_h, sup_r, sup_t], dim=1))  # (S, 3, 384)
            support_tensor = torch.stack(sup_list, dim=0)  # (B, S, 3, 384)

            logits = model(support_tensor, query_tensor)   # (B,)
            labels = torch.ones(B, device=device)      # all test triples are true

            loss = loss_fn(logits, labels)
            total_loss   += loss.item() * B
            total_samples += B

    avg_loss = total_loss / total_samples if total_samples > 0 else float("nan")
    print(f"\nBCE Loss over {total_samples:,} test triples: {avg_loss:.6f}")
    return avg_loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved GraphPFN model on test triples and report BCE loss."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the saved model checkpoint (.pt).",
    )
    parser.add_argument(
        "--train-file", type=str, required=True, metavar="TRAIN_TXT",
        help="Path to train.txt used as in-context support.",
    )
    parser.add_argument(
        "--test-file", type=str, required=True, metavar="TEST_TXT",
        help="Path to test.txt whose triples are evaluated (all labels = 1).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Number of test triples per forward pass (default: 100).",
    )
    parser.add_argument(
        "--context-size", type=int, default=None,
        help=(
            "Number of support triples sampled from train.txt per evaluation run. "
            "Should match the value used during training.  Default: use all triples."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for support sampling (default: 42).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device string (e.g. 'cpu', 'cuda', 'cuda:1'). Auto-detects if omitted.",
    )

    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from '{args.model}' (device={device}) ...")
    ckpt = torch.load(args.model, map_location=device)
    if isinstance(ckpt, dict) and "hparams" in ckpt:
        hparams = ckpt["hparams"]
        model = TriplePFN(**hparams)
        model.load_state_dict(ckpt["state_dict"])
        print(f"  Hyperparameters: {hparams}")
    else:
        # Legacy: bare state dict saved without hparams wrapper.
        model = TriplePFN()
        model.load_state_dict(ckpt)
        print("  Loaded legacy checkpoint (no hparams; using TriplePFN defaults).")

    evaluate_bce(
        model,
        train_file=args.train_file,
        test_file=args.test_file,
        batch_size=args.batch_size,
        context_size=args.context_size,
        device=device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
