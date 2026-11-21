"""Compare all-entity inference with pinned, unmodified official KGFM models.

Both implementations run sequentially in fresh processes using the same Python
environment. Dataset loading, graph construction, filtering, and fixed-walk
transfers are outside warm inference timings. Flock additionally replays the
official sampler's records through both neural models for a score parity check.
"""
import argparse
import importlib.metadata
import json
import os
import pickle
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from kgfm_inference import CHECKPOINTS, ROOT, digest

REVISIONS = {
    'ULTRA': '427966ad8ed60420eef034063d44f3153addff90',
    'TRIX': '7596e14eefefe89e61396205a0550172cadeddb0',
    'Flock': 'f35103d25a78bdf4075de5c673a51de4979aa4d7',
}
SOURCE_DIRS = {'ULTRA': 'ultra', 'TRIX': 'src', 'Flock': 'src_entity'}


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def source_record(root, official=False):
    paths = git(root, 'ls-files').splitlines()
    suffixes = {'.py', '.cpp', '.cu', '.cuh', '.h'}
    hashes = {path: digest(root / path) for path in paths if Path(path).suffix in suffixes}
    dirty = git(root, 'diff', 'HEAD', '--', *[path for path in hashes])
    if official and dirty:
        raise ValueError('The official checkout has modified source files')
    return dict(commit=git(root, 'rev-parse', 'HEAD'), source_sha256=hashes, modified=bool(dirty))


def filtered_metrics(scores, triples, directory):
    """Independent CPU pessimistic ranks; the target counts itself exactly once."""
    import torch

    with (directory / 'er_vocab.p').open('rb') as stream:
        tails = pickle.load(stream)
    with (directory / 're_vocab.p').open('rb') as stream:
        heads = pickle.load(stream)
    ranks = []
    for (h, r, t), pair in zip(triples.tolist(), scores):
        for target, values, known in [(t, pair[0], tails.get((h, r), [])),
                                      (h, pair[1], heads.get((r, t), []))]:
            eligible = values >= values[target]
            eligible[torch.tensor(list(known), dtype=torch.long)] = False
            eligible[target] = True
            ranks.append(int(eligible.sum()))
    ranks = torch.tensor(ranks, dtype=torch.float64)
    return {'MRR': ranks.reciprocal().mean().item(),
            **{f'H@{k}': (ranks <= k).double().mean().item() for k in (1, 3, 10)}}


def gpu_processes():
    if not shutil.which('nvidia-smi'):
        raise RuntimeError('nvidia-smi is required to detect competing CUDA jobs')
    run = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                          '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True)
    lines = [line for line in run.stdout.splitlines() if line.strip()]
    desktop = {'/usr/bin/kwin_wayland', '/usr/bin/kwin_x11', '/usr/lib/Xorg', '/usr/bin/Xorg'}
    other = [line for line in lines if int(line.split(',')[0]) != os.getpid()
             and line.split(',')[1].strip() not in desktop]
    if other:
        raise RuntimeError('Another CUDA process is active: ' + '; '.join(other))
    return lines


def official_model(args, facts, metadata):
    import torch
    from torch_geometric.data import Data

    source = args.upstream_root / SOURCE_DIRS[args.model]
    sys.path.insert(0, str(source.parent if args.model == 'ULTRA' else source))
    nr = metadata['num_relations']
    inverse = facts[:, [2, 1, 0]].clone()
    inverse[:, 1] += nr
    edges = torch.cat((facts, inverse)).unique(dim=0)
    graph = Data(edge_index=edges[:, [0, 2]].T, edge_type=edges[:, 1],
                 num_nodes=metadata['num_entities'], num_relations=torch.tensor(2 * nr),
                 device=torch.device(args.device))
    if args.model == 'ULTRA':
        from ultra import tasks
        from ultra.models import Ultra

        config = dict(input_dim=64, hidden_dims=[64] * 6, message_func='distmult',
                      aggregate_func='sum', short_cut=True, layer_norm=True)
        model = Ultra(dict(config, **{'class': 'RelNBFNet'}), dict(config, **{'class': 'EntityNBFNet'}))
        tasks.build_relation_graph(graph)
    elif args.model == 'TRIX':
        from trix import models_entity, tasks

        def config(n):
            return dict(input_dim=32, hidden_dims=[32] * n, message_func='distmult',
                        aggregate_func='sum', short_cut=True, layer_norm=True)
        model = models_entity.TRIX(config(3), config(2), config(4))
        tasks.build_relation_graph(graph)
    else:
        from flock.models import Flock

        model = Flock(dict(verbose=False, walk_num=args.walk_num, walk_len=128, refinements=6,
                           record_neighbors=False, attention_scatter=True, attention_scatter_n_heads=4,
                           additive_refinement=True, embed_only_first_refinement=False,
                           embedding_tying_across_refinements=False, parameter_tying_across_refinements=False,
                           net='gru', hidden_dim=64, n_layers=1, dtype='float32', test_samples=1))
    checkpoint = ROOT / 'checkpoints' / CHECKPOINTS[args.model]
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'], strict=True)
    return model, graph.to(args.device)


def worker(args):
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    source = source_record(args.upstream_root if args.variant == 'official' else ROOT,
                           official=args.variant == 'official')
    if args.variant == 'official' and source['commit'] != REVISIONS[args.model]:
        raise ValueError('Official revision differs from the pinned reference')
    metadata = json.loads((args.indexed_data / 'result.json').read_text())
    facts = torch.from_numpy(np.load(args.indexed_data / 'train_set.npy').astype(np.int64))
    test = torch.from_numpy(np.load(args.indexed_data / 'test_set.npy').astype(np.int64))
    indices = torch.randperm(len(test), generator=torch.Generator().manual_seed(args.seed))[:args.queries]
    triples_cpu = test[indices]
    triples = triples_cpu.to(args.device)
    checkpoint = ROOT / 'checkpoints' / CHECKPOINTS[args.model]
    started = time.perf_counter()
    if args.variant == 'official':
        model, graph = official_model(args, facts, metadata)
    else:
        sys.path.insert(0, str(ROOT))
        from dicee.models import TRIX, ULTRA, Flock

        cls = {'ULTRA': ULTRA, 'TRIX': TRIX, 'Flock': Flock}[args.model]
        model = cls(dict(num_entities=metadata['num_entities'], num_relations=metadata['num_relations'],
                         **{args.model.lower() + '_query_batch_size': args.query_batch_size},
                         flock_walk_num=args.walk_num, flock_test_samples=1, flock_seed=args.seed))
        model.load_pretrained(checkpoint).set_graph(facts)
    model.eval().requires_grad_(False).to(device=args.device, dtype=getattr(torch, args.dtype))
    initial_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    # Flock's official module sets matmul precision during import; override afterwards.
    torch.set_float32_matmul_precision('highest')
    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.mkldnn.matmul.fp32_precision = 'ieee'
        torch.backends.cudnn.conv.fp32_precision = 'ieee'
        torch.backends.cudnn.rnn.fp32_precision = 'ieee'
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.cuda.synchronize(args.device)
    setup_seconds = time.perf_counter() - started
    if args.dtype == 'float64':
        # Diagnostic only: official TRIX hardcodes a float32 all-ones query.
        # Promote its factory together with default boundary/edge weights, without
        # changing any arithmetic or source files. Never used for speed timings.
        torch.set_default_dtype(torch.float64)
        original_ones = torch.ones

        def double_ones(*a, **kw):
            if kw.get('dtype') == torch.float32:
                kw['dtype'] = torch.float64
            return original_ones(*a, **kw)
        torch.ones = double_ones
    records = {}
    if args.mode == 'replay':
        for i in range(len(triples)):
            for side in ('tail', 'head'):
                path = args.walks / f'{side}-{i}.pt'
                item = torch.load(path, weights_only=True, map_location=args.device)
                assert item['triple'].cpu().equal(triples_cpu[i])
                records[side, i] = tuple(item['records'])
    candidates = torch.arange(metadata['num_entities'], device=args.device)

    def predict(count=None, capture=False):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        query = triples if count is None else triples[:count]
        if args.variant == 'dice' and args.mode == 'native':
            return torch.stack((model.forward_k_vs_all(query[:, :2]),
                                model.forward_k_vs_all_heads(query[:, 1:])), 1)
        outputs = []
        for side in ('tail', 'head'):
            chunks = []
            for start in range(0, len(query), args.query_batch_size):
                chunk = query[start:start + args.query_batch_size]
                h, r, t = chunk.unbind(-1)
                if args.variant == 'official':
                    h_index = h[:, None].expand(-1, len(candidates)) if side == 'tail' else candidates.expand(len(chunk), -1)
                    t_index = candidates.expand(len(chunk), -1) if side == 'tail' else t[:, None].expand(-1, len(candidates))
                    batch = torch.stack((h_index, t_index, r[:, None].expand_as(h_index)), -1)
                    if args.mode == 'replay':
                        model.walks = lambda *a, **kw: records[side, start]
                    elif capture:
                        original_walks = model.walks

                        def record(*a, **kw):
                            value = original_walks(*a, **kw)
                            args.walks.mkdir(parents=True, exist_ok=True)
                            torch.save(dict(triple=chunk[0].cpu(), records=tuple(x.cpu() for x in value)),
                                       args.walks / f'{side}-{start}.pt')
                            return value
                        model.walks = record
                    chunks.append(model(graph, batch))
                    if capture:
                        model.walks = original_walks
                else:
                    heads = h if side == 'tail' else t
                    relations = r if side == 'tail' else r + metadata['num_relations']
                    chunks.append(model.score_walks(heads, relations, candidates.expand(len(chunk), -1), records[side, start]))
            outputs.append(torch.cat(chunks))
        return torch.stack(outputs, 1)

    def timed():
        torch.cuda.synchronize(args.device)
        begin = time.perf_counter()
        scores = predict()
        torch.cuda.synchronize(args.device)
        return time.perf_counter() - begin, scores

    snapshots = [gpu_processes()]
    if args.scores_only:
        with torch.no_grad():
            scores = predict().cpu()
        torch.save(scores, args.output / 'official-fp64.pt')
        (args.output / 'official-fp64.json').write_text(json.dumps(dict(
            source=source, benchmark_sha256=digest(Path(__file__)), checkpoint_sha256=digest(checkpoint),
            dtype=args.dtype, promoted_ones_factory=True, test_indices=indices.tolist(),
            query_batch_size=args.query_batch_size,
            metrics=filtered_metrics(scores, triples_cpu, args.indexed_data)), indent=2) + '\n')
        return
    with torch.no_grad():
        cold, scores = timed()
        del scores
        for _ in range(args.warmups):
            predict()
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        baseline_memory = torch.cuda.memory_allocated(args.device) / 2**20
        timings = []
        for _ in range(args.repeats):
            snapshots.append(gpu_processes())
            elapsed, scores = timed()
            timings.append(elapsed)
            del scores
            snapshots.append(gpu_processes())
        peak = torch.cuda.max_memory_allocated(args.device) / 2**20
        scores = predict().cpu()
        if args.variant == 'official' and args.model == 'Flock' and args.mode == 'native' and args.replay_queries:
            predict(count=min(args.replay_queries, len(triples)), capture=True)
    torch.save(scores, args.output / f'{args.variant}.pt')
    weights_unchanged = all(torch.equal(initial_weights[name], value.cpu()) for name, value in model.state_dict().items())
    if not weights_unchanged:
        raise RuntimeError('Model weights changed during inference')
    dependencies = {}
    for package in ('torch-geometric', 'torch-scatter', 'triton', 'numpy'):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    if args.variant == 'official' and args.model in ('ULTRA', 'TRIX'):
        extension = sys.modules[args.model.lower() + '.rspmm.rspmm'].rspmm
        if not hasattr(extension, 'rspmm_add_mul_forward_cuda'):
            raise RuntimeError('The official fused CUDA extension is missing')
        dependencies['rspmm_binary_sha256'] = digest(Path(extension.__file__))
    report = dict(model=args.model, variant=args.variant, mode=args.mode, source=source,
                  dataset=metadata['dataset'], test_indices=indices.tolist(), test_triples=len(triples),
                  ranked_queries=2 * len(triples), query_batch_size=args.query_batch_size,
                  walk_num=args.walk_num if args.model == 'Flock' else None, test_samples=1,
                  seed=args.seed, repeats=args.repeats, warmups=args.warmups, torch_threads=args.threads,
                  cpu_count=os.cpu_count(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                  torch=torch.__version__, cuda=torch.version.cuda, dependencies=dependencies,
                  weights_unchanged=weights_unchanged, dtype='float32', tf32=False,
                  hardware=torch.cuda.get_device_name(args.device), device=args.device,
                  benchmark_sha256=digest(Path(__file__)), checkpoint_sha256=digest(checkpoint),
                  environment={key: os.environ.get(key) for key in ['OMP_NUM_THREADS', 'CUDA_HOME', 'CC', 'CXX', 'LD_PRELOAD']},
                  train_sha256=digest(args.indexed_data / 'train_set.npy'),
                  test_sha256=digest(args.indexed_data / 'test_set.npy'), setup_seconds=setup_seconds,
                  cold_forward_seconds=cold, forward_seconds=timings,
                  median_forward_seconds=statistics.median(timings),
                  queries_per_second=2 * len(triples) / statistics.median(timings),
                  baseline_cuda_mib=baseline_memory, peak_cuda_mib=peak,
                  metrics=filtered_metrics(scores, triples_cpu, args.indexed_data),
                  gpu_process_snapshots=snapshots)
    (args.output / f'{args.variant}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['model', 'variant', 'mode', 'median_forward_seconds', 'metrics']}), flush=True)


def compare(directory, stochastic=False, fp64_path=None, raise_on_failure=True):
    import torch

    official, dice = [json.loads((directory / f'{name}.json').read_text()) for name in ('official', 'dice')]
    for key in ('test_indices', 'query_batch_size', 'walk_num', 'test_samples', 'seed', 'torch', 'cuda', 'dtype',
                'tf32', 'hardware', 'device', 'checkpoint_sha256', 'train_sha256', 'test_sha256'):
        assert official[key] == dice[key], key
    left, right = [torch.load(directory / f'{name}.pt', weights_only=True) for name in ('official', 'dice')]
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    differences = {key: dice['metrics'][key] - value for key, value in official['metrics'].items()}
    parity = torch.allclose(left, right, atol=2e-4, rtol=2e-4)
    metrics_close = all(abs(value) <= (1e-6 if key == 'MRR' else 0) for key, value in differences.items())
    result = dict(speedup=official['median_forward_seconds'] / dice['median_forward_seconds'],
                  official_seconds=official['median_forward_seconds'], dice_seconds=dice['median_forward_seconds'],
                  scores_close=bool(parity), metrics_close=metrics_close, metric_differences=differences,
                  max_absolute_score_error=(left-right).abs().max().item(),
                  independent_stochastic_walks=stochastic, validated=bool(parity and metrics_close) if not stochastic else None)
    if fp64_path is not None:
        gold_metadata = json.loads(fp64_path.with_suffix('.json').read_text())
        for key in ('checkpoint_sha256', 'test_indices', 'query_batch_size'):
            assert gold_metadata[key] == official[key], key
        gold = torch.load(fp64_path, weights_only=True)
        official_error, dice_error = (left.double()-gold).abs(), (right.double()-gold).abs()
        gold_close = torch.allclose(right.double(), gold, atol=2e-4, rtol=2e-4)
        improved = dice_error.mean() < official_error.mean()
        result['fp64_validation'] = dict(dice_scores_close=bool(gold_close), dice_more_accurate=bool(improved),
                                         official_mean_error=official_error.mean().item(), dice_mean_error=dice_error.mean().item(),
                                         official_max_error=official_error.max().item(), dice_max_error=dice_error.max().item())
        result['validated'] = bool(gold_close and improved and metrics_close)
    (directory / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if raise_on_failure and not stochastic and not result['validated']:
        raise RuntimeError('Score/metric validation failed; inspect the comparison before publishing a speedup')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=CHECKPOINTS, required=True)
    parser.add_argument('--upstream-root', type=Path, required=True)
    parser.add_argument('--indexed-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--queries', type=int, default=128)
    parser.add_argument('--query-batch-size', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--walk-num', type=int, default=128)
    parser.add_argument('--replay-queries', type=int, default=8)
    parser.add_argument('--variant', choices=['official', 'dice'], help=argparse.SUPPRESS)
    parser.add_argument('--mode', choices=['native', 'replay'], default='native', help=argparse.SUPPRESS)
    parser.add_argument('--walks', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32', help=argparse.SUPPRESS)
    parser.add_argument('--scores-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.queries, args.query_batch_size, args.repeats, args.threads, args.walk_num) < 1 or min(args.warmups, args.replay_queries) < 0:
        parser.error('Counts must be positive; warmups and replay queries may be zero')
    if not args.device.startswith('cuda'):
        parser.error('This comparison requires a CUDA device')
    if args.model == 'Flock' and args.query_batch_size != 1:
        parser.error('Use batch size 1 for the matched Flock sampling/replay protocol')
    if args.dtype != 'float32' or args.scores_only:
        if not (args.variant == 'official' and args.dtype == 'float64' and args.scores_only and args.model != 'Flock'):
            parser.error('Float64 is reserved for the official score-only diagnostic')
    for key in ('upstream_root', 'indexed_data', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    # ULTRA and TRIX both name their extension rspmm; keep their build caches separate.
    if args.model != 'Flock':
        build_root = Path(os.environ.get('TORCH_EXTENSIONS_DIR', ROOT / 'Experiments/kgfm-upstream-build'))
        os.environ['TORCH_EXTENSIONS_DIR'] = str(build_root / args.model) if not args.variant else str(build_root)
    if args.walks is None:
        args.walks = args.output / 'walks'
    if args.variant:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    modes = [('native', args.queries)]
    if args.model == 'Flock' and args.replay_queries:
        modes.append(('replay', min(args.replay_queries, args.queries)))
    for mode, queries in modes:
        directory = args.output / mode
        directory.mkdir(parents=True, exist_ok=True)
        def run_worker(variant, **overrides):
            command = [sys.executable, str(Path(__file__).resolve())]
            options = vars(args) | dict(mode=mode, queries=queries, output=directory, variant=variant) | overrides
            for key, value in options.items():
                if key == 'scores_only':
                    if value:
                        command.append('--scores-only')
                    continue
                command.extend(['--' + key.replace('_', '-'), str(value)])
            label = 'official-fp64' if options['scores_only'] else variant
            (directory / f'{label}-command.txt').write_text(shlex.join(command) + '\n')
            with (directory / f'{label}.log').open('w') as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        for variant in ('official', 'dice'):
            run_worker(variant)
        stochastic = args.model == 'Flock' and mode == 'native'
        result = compare(directory, stochastic=stochastic, raise_on_failure=False)
        if not result['scores_close'] and args.model != 'Flock':
            run_worker('official', dtype='float64', scores_only=True)
            compare(directory, fp64_path=directory / 'official-fp64.pt')
        elif not stochastic and not result['validated']:
            raise RuntimeError('Score/metric validation failed; inspect comparison.json')


if __name__ == '__main__':
    main()
