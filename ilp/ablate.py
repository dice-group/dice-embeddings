"""Ablation runner for InductiveKGModel on an inductive benchmark.

Trains the *same* `InductiveKGModel` across a grid of z_modes × tower variants,
on the same data and seed, so any metric difference is attributable to the
ablated axis alone. All arms share an identical, deterministically-built vocab.

Tower variants:
  single       candidate scored as a token in the anchor's subgraph.
  dual         candidate scored by the pooled vector of its own k-hop subgraph
               (candidate tower), anonymized independently of the anchor.
  dual_shared  dual, but the candidate subgraph is labeled from the anchor's
               [Z] assignment, so entities shared by both towers bind to the
               same slot (per-draw randomized; no persistent per-entity embed).

How the inductive setup works:

  - train on the transductive graph        `--data-dir KGs/WN18RR_v1`
    (uses its train.txt; entities here NEVER appear at inference time)
  - evaluate on a disjoint inference graph  `--ind-dir  KGs/WN18RR_v1_ind`
    whose own train.txt is the observed subgraph *context* and whose test.txt
    holds the query links to predict. Entities at inference are unseen during
    training, so the model must generalize from relational structure alone.
    The context is the inference train.txt only — test facts never enter a
    query's subgraph (each query's own triple is excluded at scoring time too).

Reuses the production training/eval core verbatim — `train.train_model` (with
auto-eval disabled), `model.load_bundle`, and `eval.{load_eval_inputs,run_eval}`
— so this is purely an experiment harness, not a reimplementation.

Run from the repo root (the dir containing the `ilp/` package):

    python -m ilp.ablate \
        --data-dir KGs/WN18RR_v1 --ind-dir KGs/WN18RR_v1_ind \
        --epochs 100 --max-triples 64 \
        --z-modes learned,runtime --variants single,dual,dual_shared \
        --json-out checkpoints/ablate/results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .eval import load_eval_inputs, print_eval_results, run_eval
from .model import load_bundle
from .train import DEFAULT_CFG, train_model


def build_cfg(args: argparse.Namespace, z_mode: str, dual: bool, shared: bool) -> dict:
    """One run's config: production defaults + this experiment's overrides.

    Everything except `z_mode`, `dual_subgraph` and `shared_anonymization` is
    held constant across the grid, and `type_relation=''` keeps the run purely
    anonymized (no [VAL_*] schema) — the regime where the tower change matters most.
    """
    cfg = dict(DEFAULT_CFG)
    cfg.update(
        data_dir=str(args.data_dir),
        triple_format=args.triple_format,
        type_relation="",           # pure anonymization — held constant
        z_mode=z_mode,
        dual_subgraph=dual,
        shared_anonymization=shared,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        max_triples=args.max_triples,
        z_pool_size=args.z_pool,
        subgraph_hops=args.subgraph_hops,
        neg_samples_per_pos=args.neg_per_pos,
        neg_sampler=args.neg_sampler,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        eval_mc=args.eval_mc,        # MC draws for runtime eval; ignored for learned
    )
    return cfg


def run_combo(args: argparse.Namespace, z_mode: str, variant: str) -> tuple[str, dict]:
    """Train one (z_mode, variant) arm, then eval it on the disjoint inference graph."""
    dual = variant in ("dual", "dual_shared")
    shared = variant == "dual_shared"
    tag = f"{z_mode}/{variant}"
    slug = f"{z_mode}_{variant}"
    cfg = build_cfg(args, z_mode, dual, shared)
    save_path = Path(args.ckpt_dir) / f"{slug}.pt"

    print(f"\n{'#' * 72}\n# TRAIN  {tag}   (dual={dual}, shared={shared}, z_mode={z_mode})\n{'#' * 72}",
          flush=True)
    t0 = time.time()
    # eval_after=False: the bundle's own data_dir is the *transductive* graph;
    # we run the inductive eval (disjoint inference graph) explicitly below instead.
    train_model(cfg, save_path=save_path, run_dir=None, eval_after=False)
    train_secs = time.time() - t0

    # Reload the self-contained bundle so eval uses exactly the trained config
    # (z_mode/dual_subgraph round-trip through cfg).
    model, vocab, fixed_values, cfg_loaded, device = load_bundle(save_path)
    fmt = args.triple_format
    ind_dir = Path(args.ind_dir)
    # Inductive eval: context = inference-graph train.txt (obs), queries =
    # inference-graph test.txt. Filter set adds valid.txt when present.
    valid_path = ind_dir / "valid.txt"
    test, known, context = load_eval_inputs(
        fmt,
        obs_files=[ind_dir / "train.txt"],
        test_files=[ind_dir / "test.txt"],
        include_test_in_context=args.context_with_test,
        filter_files=[valid_path] if valid_path.exists() else (),
    )
    print(f"\n{'=' * 72}\n= EVAL   {tag}   inductive on disjoint graph {ind_dir}\n"
          f"=   context={ind_dir / 'train.txt'}  queries={ind_dir / 'test.txt'}\n"
          f"=   #queries={len(test)}\n{'=' * 72}", flush=True)
    results = run_eval(
        model, vocab, fixed_values, cfg_loaded, device,
        test_triples=test, known_triples=known, context_triples=context,
    )
    results["_train_secs"] = train_secs
    print_eval_results(results, per_relation=args.per_relation)
    return tag, results


def print_summary(rows: list[tuple[str, dict]]) -> None:
    """Side-by-side averaged (head+tail) MRR / Hits / per-direction MRR table."""
    print(f"\n{'=' * 90}")
    print("SUMMARY — filtered metrics, averaged over head+tail (inductive)")
    print("=" * 90)
    print(f"{'arm':22s}{'MRR':>9s}{'H@1':>8s}{'H@3':>8s}{'H@10':>8s}"
          f"{'tMRR':>8s}{'hMRR':>8s}{'train_s':>10s}")
    print("-" * 90)
    for tag, r in rows:
        a = r["avg"]
        print(f"{tag:22s}{a['MRR']:>9.4f}{a['Hits@1']:>8.4f}{a['Hits@3']:>8.4f}"
              f"{a['Hits@10']:>8.4f}{r['tail']['MRR']:>8.4f}{r['head']['MRR']:>8.4f}"
              f"{r['_train_secs']:>10.1f}")
    print("=" * 90)
    print("single=candidate token; dual=independent candidate tower; "
          "dual_shared=candidate tower with anchor-shared [Z] labels.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default="KGs/WN18RR_v1",
                    help="Transductive training graph (uses its train.txt).")
    ap.add_argument("--ind-dir", default="KGs/WN18RR_v1_ind",
                    help="Disjoint inference graph: train.txt=context, test.txt=queries.")
    ap.add_argument("--triple-format", default="head_relation_tail")
    ap.add_argument("--z-modes", default="learned,runtime",
                    help="Comma-separated z_modes to sweep (learned, runtime).")
    ap.add_argument("--variants", default="single,dual,dual_shared",
                    help="Comma-separated tower variants (single, dual, dual_shared).")
    # Hyperparameters (held constant across the grid)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--max-triples", type=int, default=128)
    ap.add_argument("--z-pool", type=int, default=300)
    ap.add_argument("--subgraph-hops", type=int, default=2)
    ap.add_argument("--neg-per-pos", type=int, default=4)
    ap.add_argument("--neg-sampler", default="uniform",
                    help="uniform | relation_tail_prior | two_hop")
    ap.add_argument("--eval-mc", type=int, default=8,
                    help="MC draws for runtime-mode eval (ignored for learned).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda", help="cuda | cpu (auto-falls back).")
    ap.add_argument("--ckpt-dir", default="checkpoints/ablate",
                    help="Where per-arm .pt bundles are written.")
    ap.add_argument("--context-with-test", action="store_true",
                    help="Fold the inference test.txt into the eval context graph "
                         "(each query's own triple is still excluded).")
    ap.add_argument("--per-relation", action="store_true",
                    help="Print a per-relation MRR table after each arm.")
    ap.add_argument("--json-out", default=None,
                    help="Dump the full per-arm results dict as JSON.")
    args = ap.parse_args()

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    z_modes = [m.strip() for m in args.z_modes.split(",") if m.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = set(variants) - {"single", "dual", "dual_shared"}
    if bad:
        raise SystemExit(f"--variants must be single/dual/dual_shared, got {sorted(bad)}")

    rows: list[tuple[str, dict]] = []
    for z_mode in z_modes:
        for variant in variants:
            rows.append(run_combo(args, z_mode, variant))

    print_summary(rows)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({tag: r for tag, r in rows}, indent=2))
        print(f"\n[compare] wrote full results to {out}")


if __name__ == "__main__":
    main()
