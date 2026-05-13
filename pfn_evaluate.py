"""Evaluate a trained GraphPFN model on a test set.

Subcommands
-----------
rank (default)
    Transductive link-prediction evaluation: rank every test triple against
    all entity candidates and report MRR, MR, Hits@1/3/10.

bce
    BCE loss evaluation: report average binary cross-entropy loss over all
    test triples using entity-centric in-context support — matching the
    training distribution exactly.

Examples
--------
Ranking evaluation (MRR / Hits@k)::

    python pfn_evaluate.py rank \\
        --model model.pt \\
        --train-file KGs/Countries-S1/train.txt \\
        --test-file  KGs/Countries-S1/test.txt

Ranking evaluation with explicit context size::

    python pfn_evaluate.py rank \\
        --model model.pt \\
        --train-file KGs/UMLS/train.txt \\
        --test-file  KGs/UMLS/test.txt \\
        --context-size 64

Ranking evaluation on a specific GPU::

    python pfn_evaluate.py rank \\
        --model model.pt \\
        --train-file KGs/WN18RR/train.txt \\
        --test-file  KGs/WN18RR/test.txt \\
        --device cuda:0

BCE loss evaluation with entity-centric support::

    python pfn_evaluate.py bce \\
        --model model.pt \\
        --train-file KGs/Countries-S1/train.txt \\
        --test-file  KGs/Countries-S1/test.txt \\
        --context-size 128

BCE evaluation with a larger batch size for speed::

    python pfn_evaluate.py bce \\
        --model model.pt \\
        --train-file KGs/UMLS/train.txt \\
        --test-file  KGs/UMLS/test.txt \\
        --batch-size 256 --context-size 128

If no subcommand is given, ``rank`` is assumed.
"""

import argparse
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from pfn_dataset import _encode_strings
from pfn_inference import evaluate as evaluate_rank
from pfn_model import TriplePFN


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_model(model_path: str, device: torch.device) -> Tuple[TriplePFN, dict]:
    """Load a TriplePFN checkpoint; return (model, run_config)."""
    ckpt = torch.load(model_path, map_location=device)
    run_cfg = ckpt.get("run_config", {}) if isinstance(ckpt, dict) else {}
    if isinstance(ckpt, dict) and "hparams" in ckpt:
        model = TriplePFN(**ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
        print(f"  Hyperparameters: {ckpt['hparams']}")
    else:
        model = TriplePFN()
        model.load_state_dict(ckpt)
        print("  Loaded legacy checkpoint (no hparams; using TriplePFN defaults).")
    return model, run_cfg


def _load_triples(path: str) -> List[Tuple[str, str, str]]:
    """Read a whitespace-separated triple file; return list of (h, r, t) strings."""
    triples = []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


# ---------------------------------------------------------------------------
# BCE loss evaluation helpers
# ---------------------------------------------------------------------------

def _build_entity_index(triples: List[Tuple[str, str, str]]) -> Dict[str, List[int]]:
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

    Mirrors the entity-centric BFS used in RichSubgraphPrior:
    1. Start with the 1-hop triples of *query_h*.
    2. Expand to neighbours' triples if more slots remain.
    3. Pad with random triples if the neighbourhood is smaller than context_size.
    """
    chosen_idxs: set = set()
    support_idxs: List[int] = []

    hop1 = list(entity_index.get(query_h, []))
    rng.shuffle(hop1)
    for idx in hop1:
        if len(support_idxs) >= context_size:
            break
        if idx not in chosen_idxs:
            chosen_idxs.add(idx)
            support_idxs.append(idx)

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

    if len(support_idxs) < context_size:
        remaining = [i for i in range(len(train_triples)) if i not in chosen_idxs]
        rng.shuffle(remaining)
        for idx in remaining:
            if len(support_idxs) >= context_size:
                break
            support_idxs.append(idx)

    while len(support_idxs) < context_size:
        support_idxs.append(rng.choice(support_idxs))

    return [train_triples[i] for i in support_idxs[:context_size]]


# ---------------------------------------------------------------------------
# BCE evaluation
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

    For each test triple (h, r, t) an entity-centric support context is
    built from the neighbourhood of h in train.txt, matching the training
    distribution used by RichSubgraphPrior.

    Parameters
    ----------
    model : TriplePFN
        Trained model.
    train_file : str
        Path to train.txt used as in-context support.
    test_file : str
        Path to test.txt; every triple is treated as a positive (label=1).
    batch_size : int
        Test triples per forward pass.
    context_size : int or None
        Support triples per query. Should match the training value.
        Defaults to 32 to avoid sequence-length mismatch.
    device : torch.device or None
        Defaults to CUDA if available, else CPU.
    seed : int
        Random seed for support sampling.

    Returns
    -------
    float
        Average BCE loss over all in-vocab test triples.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

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

    entity_index = _build_entity_index(train_triples_raw)

    entity_strings   = [tok for tok, _ in sorted(entity_vocab.items(),   key=lambda x: x[1])]
    relation_strings = [tok for tok, _ in sorted(relation_vocab.items(), key=lambda x: x[1])]
    print("  Encoding strings with SentenceTransformer ...")
    entity_embs   = _encode_strings(entity_strings).to(device)
    relation_embs = _encode_strings(relation_strings).to(device)

    eff_ctx = context_size if context_size is not None else 32
    if context_size is None:
        print(f"  No --context-size given; defaulting to {eff_ctx} (set explicitly to match training).")

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
    print(f"  Evaluating {n_test:,} test triples (all labels = 1) | entity-centric support (context_size={eff_ctx}) ...")

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for start in tqdm(range(0, n_test, batch_size), desc="Evaluating", unit="batch"):
            batch = test_triples_in_vocab[start : start + batch_size]
            B = len(batch)

            q_h_embs = entity_embs[[entity_vocab[h] for h, _r, _t in batch]]
            q_r_embs = relation_embs[[relation_vocab[r] for _h, r, _t in batch]]
            q_t_embs = entity_embs[[entity_vocab[t] for _h, _r, t in batch]]
            query_tensor = torch.stack([q_h_embs, q_r_embs, q_t_embs], dim=1)

            sup_list = []
            for h_s, _r_s, _t_s in batch:
                sup_triples = _entity_centric_support(h_s, train_triples_raw, entity_index, eff_ctx, rng)
                sup_h = entity_embs[[entity_vocab[h] for h, _r, _t in sup_triples]]
                sup_r = relation_embs[[relation_vocab[r] for _h, r, _t in sup_triples]]
                sup_t = entity_embs[[entity_vocab[t] for _h, _r, t in sup_triples]]
                sup_list.append(torch.stack([sup_h, sup_r, sup_t], dim=1))
            support_tensor = torch.stack(sup_list, dim=0)

            logits = model(support_tensor, query_tensor)
            labels = torch.ones(B, device=device)

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
        description="Evaluate a trained GraphPFN model.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Common arguments factory
    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pt).")
        p.add_argument("--train-file", type=str, required=True, metavar="TRAIN_TXT", help="Path to train.txt.")
        p.add_argument("--test-file", type=str, required=True, metavar="TEST_TXT", help="Path to test.txt.")
        p.add_argument(
            "--context-size", type=int, default=None,
            help="Support triples per query. Defaults to the value stored in the checkpoint.",
        )
        p.add_argument("--device", type=str, default=None, help="Torch device string (e.g. 'cpu', 'cuda').")

    # ------------------------------------------------------------------
    # rank subcommand — MRR / Hits@k
    # ------------------------------------------------------------------
    rank_parser = subparsers.add_parser(
        "rank",
        help="Transductive link-prediction: MRR, MR, Hits@1/3/10.",
        description="Rank every test triple against all entity candidates and report ranking metrics.",
    )
    _add_common(rank_parser)

    # ------------------------------------------------------------------
    # bce subcommand — BCE loss
    # ------------------------------------------------------------------
    bce_parser = subparsers.add_parser(
        "bce",
        help="BCE loss evaluation with entity-centric in-context support.",
        description=(
            "Compute average binary cross-entropy loss over test triples.\n"
            "In-context support is entity-centric (matches training distribution)."
        ),
    )
    _add_common(bce_parser)
    bce_parser.add_argument("--batch-size", type=int, default=100, help="Test triples per forward pass.")
    bce_parser.add_argument("--seed", type=int, default=42, help="Random seed for support sampling.")

    # Default to rank if no subcommand given
    if len(sys.argv) == 1 or sys.argv[1] not in ("rank", "bce", "-h", "--help"):
        sys.argv.insert(1, "rank")

    args = parser.parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from '{args.model}' (device={device}) ...")
    model, run_cfg = _load_model(args.model, device)

    # Resolve context_size: CLI > checkpoint > fallback
    context_size = args.context_size
    if context_size is None and run_cfg:
        context_size = int(run_cfg.get("context_size", 128))
        print(f"Using checkpoint context_size={context_size}.")

    # ------------------------------------------------------------------
    # rank
    # ------------------------------------------------------------------
    if args.command == "rank":
        model.to(device)
        model.eval()
        metrics = evaluate_rank(
            model,
            train_file=args.train_file,
            test_file=args.test_file,
            device=device,
            context_size=context_size,
        )
        print("\n" + "=" * 40)
        print("Link-Prediction Results")
        print("=" * 40)
        print(f"  MRR:     {metrics['MRR']:.4f}")
        print(f"  MR:      {metrics['MR']:.1f}")
        print(f"  Hits@1:  {metrics['Hits@1']:.4f}")
        print(f"  Hits@3:  {metrics['Hits@3']:.4f}")
        print(f"  Hits@10: {metrics['Hits@10']:.4f}")

    # ------------------------------------------------------------------
    # bce
    # ------------------------------------------------------------------
    elif args.command == "bce":
        evaluate_bce(
            model,
            train_file=args.train_file,
            test_file=args.test_file,
            batch_size=args.batch_size,
            context_size=context_size,
            device=device,
            seed=args.seed,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
