"""Benchmark KGFM inference against a source checkout, with score/metric checks.

Uses indexed splits emitted by kgfm_zero_shot.py. Each implementation runs in
its own subprocess. Timings synchronize CUDA; compilation/layout construction
is reported separately from warm steady-state inference and evaluation.
"""
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {'ULTRA': 'ultra_3g.pth', 'TRIX': 'trix/entity_prediction.pth', 'Flock': 'flock/flock_entity.pth'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def worker(args):
    sys.path.insert(0, str(args.source_root))
    import numpy as np
    import torch

    from dicee.evaluation.link_prediction import evaluate_lp
    from dicee.models import TRIX, ULTRA, Flock

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    directory = args.indexed_data
    metadata = json.loads((directory / 'result.json').read_text())
    facts = np.load(directory / 'train_set.npy')
    triples = np.load(directory / 'test_set.npy')[:args.queries]
    if not len(triples):
        raise ValueError('No test queries')
    import pickle
    with (directory / 'er_vocab.p').open('rb') as stream:
        er = pickle.load(stream)
    with (directory / 're_vocab.p').open('rb') as stream:
        re = pickle.load(stream)
    checkpoint = ROOT / 'checkpoints' / CHECKPOINTS[args.model]
    settings = dict(num_entities=metadata['num_entities'], num_relations=metadata['num_relations'],
                    **{args.model.lower() + '_query_batch_size': args.query_batch_size},
                    flock_walk_num=128, flock_test_samples=1, flock_seed=42)
    cls = {'ULTRA': ULTRA, 'TRIX': TRIX, 'Flock': Flock}[args.model]
    begin = time.perf_counter()
    model = cls(settings).load_pretrained(checkpoint).set_graph(facts).eval().requires_grad_(False).to(device=args.device, dtype=getattr(torch, args.dtype))
    setup_seconds = time.perf_counter() - begin
    query = torch.as_tensor(triples.astype(np.int64))

    def synchronize():
        if args.device.startswith('cuda'):
            torch.cuda.synchronize(args.device)

    def predict():
        return torch.stack((model.forward_k_vs_all(query[:, :2]), model.forward_k_vs_all_heads(query[:, 1:])), 1)

    def timed(fn):
        synchronize()
        start = time.perf_counter()
        value = fn()
        synchronize()
        return time.perf_counter() - start, value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.scores_only:
        with torch.no_grad():
            torch.save(predict().cpu(), args.output.with_suffix('.pt'))
        args.output.write_text(json.dumps(dict(source_root=str(args.source_root), dtype=args.dtype,
                                               checkpoint_sha256=digest(checkpoint), settings=settings)) + '\n')
        return

    with torch.no_grad():
        cold, output = timed(predict)
        del output
        for _ in range(args.warmups):
            predict()
        synchronize()
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(args.device)
        timings = []
        for _ in range(args.repeats):
            elapsed, output = timed(predict)
            timings.append(elapsed)
            del output
        peak = torch.cuda.max_memory_allocated(args.device) / 2**20 if args.device.startswith('cuda') else None
        scores = predict().cpu()
        evaluation = []
        for _ in range(args.repeats):
            elapsed, metrics = timed(lambda: evaluate_lp(model, triples, metadata['num_entities'], er, re,
                                                         batch_size=args.batch_size, tie_policy=args.tie_policy, tie_seed=42))
            evaluation.append(elapsed)
    torch.save(scores, args.output.with_suffix('.pt'))
    sources = {str(p.relative_to(args.source_root)): digest(p) for p in sorted((args.source_root / 'dicee').rglob('*.py'))}
    report = dict(model=args.model, dataset=metadata['dataset'], test_triples=len(triples), ranked_queries=2 * len(triples),
                  settings=settings, source_root=str(args.source_root), source_sha256=sources,
                  checkpoint_sha256=digest(checkpoint), train_sha256=digest(directory/'train_set.npy'),
                  test_sha256=digest(directory/'test_set.npy'), torch=torch.__version__, cuda=torch.version.cuda,
                  device=args.device, hardware=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'CPU',
                  dtype=args.dtype, tf32=False, threads=args.threads, batch_size=args.batch_size, tie_policy=args.tie_policy,
                  repeats=args.repeats, warmups=args.warmups, setup_seconds=setup_seconds, cold_forward_seconds=cold,
                  forward_seconds=timings, median_forward_seconds=statistics.median(timings),
                  evaluation_seconds=evaluation, median_evaluation_seconds=statistics.median(evaluation),
                  queries_per_second=2*len(triples)/statistics.median(timings), peak_cuda_allocated_mib=peak, metrics=metrics)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['model', 'dataset', 'median_forward_seconds', 'queries_per_second',
                                                'median_evaluation_seconds', 'peak_cuda_allocated_mib', 'metrics']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--indexed-data', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--model', choices=CHECKPOINTS, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--query-batch-size', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--queries', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--tie-policy', choices=['sort', 'optimistic', 'random', 'pessimistic'], default='sort')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=ROOT, help=argparse.SUPPRESS)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32', help=argparse.SUPPRESS)
    parser.add_argument('--scores-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.query_batch_size, args.batch_size, args.queries, args.repeats, args.threads) < 1 or args.warmups < 0:
        parser.error('Counts must be positive; warmups may be zero')
    if args.worker:
        worker(args)
        return
    report = {}
    variants = [('reference', args.reference_root), ('optimized', ROOT)] if args.reference_root else [('optimized', ROOT)]
    for label, source in variants:
        output = args.output / (label + '.json')
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--source-root', str(source.resolve()),
                   '--indexed-data', str(args.indexed_data.resolve()), '--output', str(output.resolve())]
        for key in ['model', 'device', 'query_batch_size', 'batch_size', 'queries', 'repeats', 'warmups', 'threads', 'tie_policy']:
            command.extend(['--' + key.replace('_', '-'), str(getattr(args, key))])
        subprocess.run(command, check=True)
        report[label] = json.loads(output.read_text())
    if args.reference_root:
        import torch
        reference = torch.load(args.output/'reference.pt', weights_only=True)
        optimized = torch.load(args.output/'optimized.pt', weights_only=True)
        delta = (reference - optimized).abs()
        score_parity = torch.allclose(reference, optimized, atol=2e-4, rtol=2e-4)
        metric_deltas = {key: report['optimized']['metrics'][key] - value for key, value in report['reference']['metrics'].items()}
        report['comparison'] = dict(forward_speedup=report['reference']['median_forward_seconds']/report['optimized']['median_forward_seconds'],
                                    evaluation_speedup=report['reference']['median_evaluation_seconds']/report['optimized']['median_evaluation_seconds'],
                                    max_absolute_score_error=delta.max().item(), mean_absolute_score_error=delta.mean().item(),
                                    scores_close=score_parity, atol=2e-4, rtol=2e-4, metric_deltas=metric_deltas)
        report['comparison']['validated'] = score_parity
        if not score_parity and args.model in ('ULTRA', 'TRIX'):
            # Large float32 sums in the old scatter kernel can be less accurate
            # than the fused tree reduction. Resolve that case against the old
            # implementation in float64, without relaxing the score tolerance.
            gold_command = list(command)
            gold_command[gold_command.index('--source-root') + 1] = str(args.reference_root.resolve())
            gold_command[gold_command.index('--output') + 1] = str((args.output/'reference-fp64.json').resolve())
            subprocess.run(gold_command + ['--dtype', 'float64', '--scores-only'], check=True)
            gold = torch.load(args.output/'reference-fp64.pt', weights_only=True)
            optimized_error = (optimized.double() - gold).abs()
            reference_error = (reference.double() - gold).abs()
            gold_close = torch.allclose(optimized.double(), gold, atol=2e-4, rtol=2e-4)
            improved = bool(optimized_error.mean() < reference_error.mean())
            # A precision correction must also preserve reported hit rates and
            # keep the sampled MRR difference below one part per million.
            metrics_close = all(abs(delta) <= (1e-6 if key == 'MRR' else 0) for key, delta in metric_deltas.items())
            report['comparison']['fp64_validation'] = dict(
                optimized_scores_close=gold_close, precision_improved=improved,
                optimized_max_error=optimized_error.max().item(), optimized_mean_error=optimized_error.mean().item(),
                reference_max_error=reference_error.max().item(), reference_mean_error=reference_error.mean().item(),
                metrics_close=metrics_close)
            report['comparison']['validated'] = gold_close and improved and metrics_close
        print(json.dumps(report['comparison']), flush=True)
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.reference_root and not report['comparison']['validated']:
        raise RuntimeError('Score parity failed; inspect comparison.json')


if __name__ == '__main__':
    main()
