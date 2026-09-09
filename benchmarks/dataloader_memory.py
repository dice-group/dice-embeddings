"""Measure private worker RAM over one synthetic dataset pass, without training.

Example: python benchmarks/dataloader_memory.py --dataset KvsAll --start-method fork
Report USS (unique/private memory), not summed RSS which double-counts shared pages.
"""
# ruff: noqa: E402
# The source checkout must precede installed packages when run as a script.
import argparse
import gc
import json
import multiprocessing
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import torch

from dicee.dataset_classes._bpe import MultiLabelDataset
from dicee.dataset_classes._label_based import AllvsAll, FSDP1vsSampleDataset, KvsAll, KvsSampleDataset
from dicee.dataset_classes._negative_sampling import GroupedNegativeSamplingDataset
from dicee.dataset_classes._storage import RaggedIndices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['KvsAll', 'AllvsAll', 'KvsSample', 'BPE', 'Strict', 'FSDP'], default='KvsAll')
    parser.add_argument('--pairs', type=int, default=200_000)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--start-method', choices=multiprocessing.get_all_start_methods(), default=multiprocessing.get_start_method())
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.pairs < 1 or args.workers < 1 or args.batch_size < 1:
        parser.error('pairs, workers, and batch-size must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    indices = np.arange(args.pairs, dtype=np.int64)
    facts = np.empty((args.pairs * 2, 3), dtype=np.int64)
    facts[:, 0] = np.repeat(indices % 64, 2)
    facts[:, 1] = np.repeat(indices // 64, 2)
    facts[:, 2] = np.tile(np.array([0, 1], dtype=np.int64), args.pairs)
    entities = range(128 if args.dataset == 'Strict' else 64)
    relations = range((args.pairs + 63) // 64)
    if args.dataset == 'KvsAll':
        dataset = KvsAll(facts, entities, relations, 'EntityPrediction')
    elif args.dataset == 'AllvsAll':
        dataset = AllvsAll(facts, entities, relations)
    elif args.dataset == 'KvsSample':
        dataset = KvsSampleDataset(facts, entities, relations, 'EntityPrediction', neg_ratio=3)
    elif args.dataset == 'BPE':
        targets = RaggedIndices(facts[:, 2].copy(), np.arange(args.pairs + 1) * 2)
        dataset = MultiLabelDataset(torch.zeros(args.pairs, 2, 2, dtype=torch.long), targets,
                                    len(entities), torch.zeros(len(entities), 2, dtype=torch.long))
    elif args.dataset == 'Strict':
        dataset = GroupedNegativeSamplingDataset(facts, len(entities), len(relations),
                                                  neg_sample_ratio=3, strict_negative_sampling=True)
    else:
        dataset = FSDP1vsSampleDataset(facts, entities, relations, 'EntityPrediction', neg_ratio=3)
    del facts, indices
    gc.collect()
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, persistent_workers=True,
                                         prefetch_factor=2, collate_fn=dataset.collate_fn,
                                         multiprocessing_context=args.start_method)
    iterator = iter(loader)
    workers = [psutil.Process(worker.pid) for worker in iterator._workers]
    snapshots = []

    def snapshot(samples):
        return {'samples': samples, 'worker_uss_mib': [round(worker.memory_full_info().uss / 2**20, 2) for worker in workers]}

    start = time.monotonic()
    try:
        for batch, _ in enumerate(iterator):
            if batch in {min(3, len(loader) - 1), len(loader) // 2}:
                snapshots.append(snapshot(min((batch + 1) * args.batch_size, len(dataset))))
        snapshots.append(snapshot(len(dataset)))
    finally:
        iterator._shutdown_workers()
    result = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'python': platform.python_version(), 'torch': torch.__version__,
        'seconds': round(time.monotonic() - start, 2), 'snapshots': snapshots,
        'worker_uss_growth_mib': [round(end - begin, 2) for begin, end in
                                zip(snapshots[0]['worker_uss_mib'], snapshots[-1]['worker_uss_mib'])],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
