"""Run frozen KGFM entity evaluation with DICE's existing loader and ranking.

Example: python benchmarks/kgfm_zero_shot.py --model Flock --dataset Countries-S1
Progress is saved after each evaluation batch; identical runs resume safely.
"""
# ruff: noqa: E402
# The source checkout must precede installed packages when run as a script.
import argparse
import hashlib
import json
import logging
import os
import platform
import resource
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from dicee.evaluation._filtering import TIE_POLICIES
from dicee.evaluation.link_prediction import evaluate_lp
from dicee.knowledge_graph import KG
from dicee.models import TRIX, ULTRA, Flock

MODELS = {"ULTRA": ULTRA, "TRIX": TRIX, "Flock": Flock}
CHECKPOINTS = {
    "ULTRA": "checkpoints/ultra_3g.pth",
    "TRIX": "checkpoints/trix/entity_prediction.pth",
    "Flock": "checkpoints/flock/flock_entity.pth",
}
METRICS = ("MRR", "H@1", "H@3", "H@10")


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--query-batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--walk-num", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tie-policy", choices=TIE_POLICIES, default="sort")
    parser.add_argument("--tie-seed", type=int, default=None,
                        help="Independent random tie seed; defaults to --seed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tie_seed is None:
        args.tie_seed = args.seed
    if min(args.batch_size, args.query_batch_size, args.threads, args.walk_num, args.test_samples) < 1:
        parser.error("Batch sizes, threads, and sampling counts must be positive")
    os.chdir(ROOT)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    logging.basicConfig(level=logging.WARNING)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_dir = ROOT / "KGs" / args.dataset
    checkpoint = ROOT / CHECKPOINTS[args.model]
    sources = sorted((ROOT / "dicee").rglob("*.py")) + [Path(__file__).resolve()]
    cpu_name = next((line.split(":", 1)[1].strip() for line in
                     Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown")
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": digest(checkpoint),
        "split_sha256": {name: digest(dataset_dir / f"{name}.txt") for name in ("train", "valid", "test")},
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sources},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__, "python": platform.python_version(),
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else cpu_name,
        "cpu": cpu_name, "dtype": "float32", "tf32": False,
        "num_epochs": 0, "optimizer_updates": 0,
        "inference_edges": "train plus generated inverses, deduplicated",
        "filter_positives": ["train", "valid", "test"],
        "candidates": "all entities in the combined split vocabulary",
        "directions": ["head", "tail"], "tie_policy": args.tie_policy,
        "flock_walk_len": 128, "flock_refinements": 6,
    }
    config_path = output / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError(f"Configuration/source changed; use a new output directory: {output}")
    write_json(config_path, config)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    if (output / "result.json").exists():
        print((output / "result.json").read_text(), flush=True)
        return
    start = time.perf_counter()
    kg = KG(dataset_dir=str(dataset_dir), add_reciprocal=False, eval_model="test",
            training_technique="NegSample", separator=r"\s+", backend="pandas",
            path_for_serialization=str(output))
    if kg.valid_set is None or kg.test_set is None or not len(kg.test_set):
        raise ValueError("The benchmark requires nonempty test and available validation splits")
    # Refuse to publish a held-out score if a test fact is already in context.
    train_facts = set(map(tuple, kg.train_set.tolist()))
    if any(tuple(triple) in train_facts for triple in kg.test_set.tolist()):
        raise ValueError("Training and test facts overlap")
    del train_facts
    settings = dict(num_entities=kg.num_entities, num_relations=kg.num_relations,
                    **{f"{args.model.lower()}_query_batch_size": args.query_batch_size},
                    flock_walk_num=args.walk_num, flock_test_samples=args.test_samples,
                    flock_seed=args.seed)
    model = MODELS[args.model](settings).load_pretrained(checkpoint)
    model.set_graph(kg.train_set).eval().requires_grad_(False)
    initial_weights = {key: value.clone() for key, value in model.state_dict().items()}
    model.to(device)
    er_vocab = kg.er_vocab if isinstance(kg.er_vocab, dict) else kg.er_vocab.result()
    re_vocab = kg.re_vocab if isinstance(kg.re_vocab, dict) else kg.re_vocab.result()
    # Persist filter vocabularies to permit independent evaluation of the run.
    import pickle
    for name, vocab in (("er_vocab", er_vocab), ("re_vocab", re_vocab)):
        with (output / f"{name}.p").open("wb") as stream:
            pickle.dump(vocab, stream)
    progress_path = output / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    batches = progress.get("batches", [])
    tie_generator = None
    if args.tie_policy == "random":
        tie_generator = torch.Generator(device="cpu").manual_seed(args.tie_seed)
        if batches:
            tie_generator.set_state(torch.tensor(progress["tie_rng_state"], dtype=torch.uint8))
    completed = sum(batch["triples"] for batch in batches)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    print(f"START {args.model} {args.dataset}: {completed}/{len(kg.test_set)} test triples; {device}", flush=True)
    with torch.no_grad():
        for offset in range(completed, len(kg.test_set), args.batch_size):
            triples = kg.test_set[offset:offset + args.batch_size]
            batch_start = time.perf_counter()
            scores = evaluate_lp(model, triples, kg.num_entities, er_vocab, re_vocab,
                                 batch_size=args.batch_size, tie_policy=args.tie_policy,
                                 tie_seed=args.tie_seed, tie_generator=tie_generator)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batches.append({"offset": offset, "triples": len(triples), "metrics": scores,
                            "seconds": time.perf_counter() - batch_start})
            completed += len(triples)
            progress = {"completed": completed, "total": len(kg.test_set), "batches": batches}
            if tie_generator is not None:
                progress["tie_rng_state"] = tie_generator.get_state().tolist()
            write_json(progress_path, progress)
            print(f"PROGRESS {args.model} {args.dataset}: {completed}/{len(kg.test_set)}", flush=True)
    for key, value in model.state_dict().items():
        if not torch.equal(value.cpu(), initial_weights[key]):
            raise AssertionError(f"Frozen weights changed: {key}")
    metrics = {key: sum(batch["metrics"][key] * batch["triples"] for batch in batches) / len(kg.test_set)
               for key in METRICS}
    result = {
        "status": "complete", "dataset": args.dataset, "model": args.model,
        "target_graph_in_pretraining": args.dataset in ("FB15k-237", "WN18RR"),
        "test_triples": len(kg.test_set), "ranked_queries": 2 * len(kg.test_set),
        "num_entities": kg.num_entities, "num_relations": kg.num_relations,
        "inference_edges": model.edge_type.numel(), "metrics": metrics,
        "evaluation_seconds": sum(batch["seconds"] for batch in batches),
        "invocation_seconds": time.perf_counter() - start,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "weights_unchanged": True, "configuration": config,
    }
    write_json(output / "eval_report.json", {"Test": metrics})
    write_json(output / "result.json", result)
    print(f"COMPLETE {args.model} {args.dataset}: {json.dumps(metrics)}", flush=True)


if __name__ == "__main__":
    main()
