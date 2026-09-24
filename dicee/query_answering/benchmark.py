"""Streaming, full-candidate evaluation of published logical queries.

The default sort/filter/query/type averaging protocol follows UltraQuery's
query_utils.py at 427966ad8ed60420eef034063d44f3153addff90. Predictions are
provided independently of labels so any query executor can use this evaluator.
"""

import hashlib
import heapq
import json
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch

from ..evaluation._filtering import FilteredRanker
from ._checkpoint import BenchmarkCheckpoint, write_json
from ._query import QUERY_SHAPES, compile_query, relation_signature
from .adapter import QueryScoreAdapter
from .context import attached_context, evaluation_mode, fingerprint, state_fingerprint
from .datasets import BENCHMARK_DATASETS, load_benchmark
from .engine import QueryAnswerer

METRICS = ('mrr', 'hits1', 'hits3', 'hits10')


class QueryMetrics:
    """Reuse a validated candidate domain and device masks across a dataset."""

    def __init__(self, num_entities, *, candidates=None, tie_policy='sort'):
        if tie_policy not in ('sort', 'average', 'optimistic', 'pessimistic'):
            raise ValueError('Unknown benchmark tie policy')
        self.n, self.tie_policy = num_entities, tie_policy
        domain = range(num_entities) if candidates is None else frozenset(candidates)
        if not domain or min(domain) < 0 or max(domain) >= num_entities:
            raise ValueError('Invalid candidate domain')
        self.full_domain = len(domain) == num_entities
        self.domain = range(num_entities) if self.full_domain else domain
        self.outside = frozenset() if self.full_domain else frozenset(range(num_entities)) - domain
        self._device = None

    def __call__(self, scores, easy, hard):
        if scores.ndim != 1 or len(scores) != self.n or (torch.isnan(scores) | torch.isposinf(scores)).any():
            raise ValueError('Expected one complete higher-is-better score vector')
        easy, hard = frozenset(easy), frozenset(hard)
        answers = easy | hard
        if not hard or easy & hard or any(t not in self.domain for t in answers):
            raise ValueError('Need disjoint easy/hard answers and nonempty hard answers inside the candidate domain')
        targets = sorted(hard)
        if self.tie_policy == 'sort':
            if self._device != scores.device:
                self._negative_template = torch.ones(self.n, dtype=torch.long, device=scores.device)
                self._outside_index = torch.tensor(sorted(self.outside), dtype=torch.long, device=scores.device)
                self._negative_template[self._outside_index] = 0
                self._positions = torch.arange(self.n, device=scores.device)
                self._device = scores.device
            # Sort before filtering, including outside-domain -inf entries, to
            # preserve the reference evaluator's tie ordering exactly.
            values = scores if scores.is_contiguous() else scores.clone()
            if not self.full_domain:
                values = scores.clone()
                values[self._outside_index] = -torch.inf
            order = values.argsort(descending=True)
            negatives = self._negative_template.clone()
            negatives[sorted(answers)] = 0
            positions = torch.empty_like(order)
            positions[order] = self._positions
            ranks = 1 + negatives[order].cumsum(0)[positions[targets]]
        else:
            ranker = FilteredRanker('optimistic')
            excluded = sorted(self.outside | answers)
            bounds = [ranker.bounds_batch(scores[None], [target], [excluded])[0] for target in targets]
            factor = {'average': .5, 'optimistic': 0., 'pessimistic': 1.}[self.tie_policy]
            ranks = torch.tensor([rank + factor * tied for rank, tied in bounds], dtype=torch.float64)
        ranks = ranks.to(torch.float64)
        metrics = torch.stack([ranks.reciprocal().mean(), *((ranks <= k).double().mean() for k in (1, 3, 10))])
        return dict(zip(METRICS, metrics.tolist()))


def filtered_query_metrics(scores, easy, hard, *, candidates=None, tie_policy='sort'):
    """Filter other true answers and average over hard answers for one query.

    ``sort`` preserves PyTorch's descending argsort tie ordering, as in the
    reference evaluator. Reuse QueryMetrics when evaluating a whole dataset.
    """
    if scores.ndim != 1:
        raise ValueError('Expected one complete higher-is-better score vector')
    return QueryMetrics(len(scores), candidates=candidates, tie_policy=tie_policy)(scores, easy, hard)


def _averages(per_shape):
    result = {}
    for name, shapes in (('all', list(per_shape)), ('epfo', [s for s in per_shape if 'n' not in s]),
                        ('negation', [s for s in per_shape if 'n' in s])):
        result[name] = ({metric: sum(per_shape[s][metric] for s in shapes) / len(shapes) for metric in METRICS}
                        | {'shapes': len(shapes)}) if shapes else None
    return result


def _query_plan(data, limit, order, *, sampling='prefix', seed=0):
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('Query limit must be a positive integer')
    if order not in ('published', 'relation'):
        raise ValueError('Unknown query order')
    if sampling not in ('prefix', 'uniform') or type(seed) is not int:
        raise ValueError('Choose prefix/uniform sampling and an integer sampling seed')
    counts, selected = Counter(), []
    if sampling == 'uniform' and limit is not None:
        groups = defaultdict(list)
        for query in data.queries:
            groups[query.shape].append(query)
        for shape, queries in groups.items():
            def priority(query):
                return fingerprint(('query-sample-v1', seed, data.name, data.split, shape, query.query)), query.query
            selected.extend(heapq.nsmallest(limit, queries, key=priority))
        selected.sort(key=lambda q: (q.shape, q.query))
    else:
        for query in data.queries:
            if limit is None or counts[query.shape] < limit:
                selected.append(query)
                counts[query.shape] += 1
    if order == 'relation':
        selected.sort(key=lambda q: (q.shape, relation_signature(compile_query(q.query)), q.query))
    return selected


def _plan_fingerprint(queries):
    digest = hashlib.sha256()
    for query in queries:
        digest.update(fingerprint((query.shape, query.query, sorted(query.easy), sorted(query.hard))).encode())
    return digest.hexdigest()


def _implementation_fingerprint():
    root = Path(__file__).parents[1]
    paths = sorted(Path(__file__).parent.glob('*.py')) + sorted((root / 'models').glob('*.py'))
    paths.append(root / 'evaluation' / '_filtering.py')
    return fingerprint({str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})


@torch.no_grad()
def evaluate_benchmark(data, predict, *, tie_policy='sort', max_queries_per_shape=None, progress=None,
                       prepare=None, query_batch_size=1, query_order='published', checkpoint_dir=None,
                       checkpoint_identity=None, checkpoint_every=500, on_checkpoint=None, statistics=None,
                       query_sampling='prefix', sampling_seed=0, on_query=None):
    """Stream filtered metrics, optionally resuming a verified query prefix."""
    if type(query_batch_size) is not int or query_batch_size < 1 or type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError('Batch and checkpoint intervals must be positive integers')
    selected = _query_plan(data, max_queries_per_shape, query_order, sampling=query_sampling, seed=sampling_seed)
    metrics_for_query = QueryMetrics(data.context.num_entities, candidates=data.candidates, tie_policy=tie_policy)
    if checkpoint_dir is not None and checkpoint_identity is None:
        raise ValueError('Checkpointing requires an explicit predictor identity')
    identity = dict(predictor=checkpoint_identity, implementation=_implementation_fingerprint(),
                    dataset=data.name, split=data.split, context=data.context.identity, metadata=data.metadata,
                    candidates=fingerprint(data.candidates), plan=_plan_fingerprint(selected),
                    tie_policy=tie_policy, limit=max_queries_per_shape, order=query_order,
                    sampling=query_sampling, sampling_seed=sampling_seed,
                    batch=query_batch_size, checkpoint_every=checkpoint_every) if checkpoint_dir is not None else None
    store = BenchmarkCheckpoint(checkpoint_dir, identity) if checkpoint_dir is not None else nullcontext(None)
    available = Counter(q.shape for q in data.queries)
    stats = Counter() if statistics is None else statistics
    with store as checkpoint:
        saved = checkpoint.load() if checkpoint else None
        counts, hard_counts, sums = Counter(), Counter(), defaultdict(Counter)
        completed, elapsed = 0, 0.
        if saved is not None:
            completed, elapsed = saved['completed'], saved['seconds']
            if not 0 <= completed <= len(selected):
                raise ValueError('Invalid checkpoint query count')
            counts.update(saved['counts'])
            hard_counts.update(saved['hard_counts'])
            sums.update({shape: Counter(values) for shape, values in saved['sums'].items()})
            stats.update(saved['statistics'])
            if sum(counts.values()) != completed:
                raise ValueError('Invalid checkpoint shape counts')
        started = time.monotonic()

        def report():
            per_shape = {shape: {metric: sums[shape][metric] / counts[shape] for metric in METRICS}
                         | dict(queries=counts[shape], hard_answers=hard_counts[shape], available_queries=available[shape])
                         for shape in sorted(counts)}
            return dict(dataset=data.name, group=data.group, split=data.split, per_shape=per_shape, averages=_averages(per_shape),
                        queries=completed, seconds=elapsed + time.monotonic() - started,
                        protocol=dict(tie_policy=tie_policy, filtering='all other easy and hard answers',
                                      averaging='answers per query, queries per shape, equal shapes per group',
                                      max_queries_per_shape=max_queries_per_shape, query_order=query_order,
                                      query_sampling=query_sampling, sampling_seed=sampling_seed,
                                      full_split=completed == len(data.queries), all_14_shapes=set(counts) == set(QUERY_SHAPES)),
                        dataset_metadata=data.metadata, context_sha256=data.context.identity,
                        candidate_sha256=fingerprint(data.candidates), num_candidates=len(data.candidates))

        if progress and completed:
            progress(completed, len(selected))
        while completed < len(selected):
            end = min(completed + query_batch_size, len(selected), (completed // checkpoint_every + 1) * checkpoint_every)
            batch = selected[completed:end]
            if prepare:
                prepare([q.query for q in batch])
            for q in batch:
                scores = predict(q.query)
                if scores.shape != (data.context.num_entities,):
                    raise ValueError('Predictor must score the complete public entity vocabulary')
                metrics = metrics_for_query(scores, q.easy, q.hard)
                if on_query:
                    on_query(q.query, q.shape, dict(metrics))
                counts[q.shape] += 1
                hard_counts[q.shape] += len(q.hard)
                sums[q.shape].update(metrics)
                completed += 1
                if completed % checkpoint_every == 0 or completed == len(selected):
                    current = report()
                    if checkpoint:
                        checkpoint.save(dict(completed=completed, seconds=current['seconds'], counts=dict(counts),
                                             hard_counts=dict(hard_counts), sums=dict(sums), statistics=dict(stats)))
                    if on_checkpoint:
                        on_checkpoint(current)
                if progress:
                    progress(completed, len(selected))
        return report()


def benchmark_model(model, data, *, adapter=None, observed_mix=None, beam_size=64, tnorm='prod',
                    row_batch_size=8, backend_batch_size=None, cache_bytes=512 * 1024 * 1024, seed=0, samples=None,
                    tie_policy='sort', max_queries_per_shape=None, progress=None, query_batch_size=32,
                    query_order='relation', checkpoint_dir=None, checkpoint_every=500, on_checkpoint=None,
                    query_sampling='prefix', sampling_seed=0, on_query=None):
    """Run CQD with shared caching, batching and durable progress across backends."""
    from ..models._inference import float32_precision_token
    from ..models.flock import Flock
    previous_batch = getattr(model, 'query_batch_size', None)
    effective_batch = backend_batch_size
    if effective_batch is None:
        effective_batch = previous_batch if isinstance(model, Flock) else row_batch_size
    if type(effective_batch) is not int or effective_batch < 1:
        raise ValueError('Backend batch size must be a positive integer')
    stats = Counter()
    try:
        if previous_batch is not None:
            model.query_batch_size = effective_batch
        with attached_context(model, data.context), evaluation_mode(model):
            engine = QueryAnswerer(model, context=data.context, adapter=adapter, observed_mix=observed_mix,
                                   row_batch_size=row_batch_size, cache_bytes=cache_bytes, seed=seed, samples=samples)
            transform = engine.adapter
            inference = dict(method='cqd-global-prefix', model=model.name, beam_size=beam_size, tnorm=tnorm,
                             negation='standard', score_space='float64-log-memberships',
                             row_batch_size=row_batch_size, backend_batch_size=effective_batch if previous_batch is not None else None,
                             query_batch_size=query_batch_size, cache_bytes=cache_bytes, cache_scope='evaluation-session',
                             seed=seed, samples=samples,
                             effective_samples=getattr(model, 'test_samples', None) if samples is None else samples,
                             sampling='independent-row-single-sample-v1' if isinstance(model, Flock) else None,
                             adapter=transform.to_dict(),
                             backbone_state_sha256=state_fingerprint(model), device=str(engine.scorer.device),
                             cuda=torch.version.cuda,
                             gpu=torch.cuda.get_device_name(engine.scorer.device) if engine.scorer.device.type == 'cuda' else None,
                             torch=str(torch.__version__), float32_precision=str(float32_precision_token()),
                             cpu_threads=torch.get_num_threads(),
                             deterministic=torch.are_deterministic_algorithms_enabled(),
                             model_configuration={k: str(v) for k, v in getattr(model, 'args', {}).items()},
                             runtime_configuration={k: getattr(model, k, None) for k in (
                                 '_inference_backend', 'query_batch_size', 'relation_cache_mb', 'projection_cache_mb',
                                 'walk_num', 'walk_len', 'refinements', 'compact_state', 'pack_walks', 'compile_sampler', 'prefetch_walks')},
                             module_configuration={name: {k: getattr(module, k, None) for k in ('inference_backend', 'inference_compile')}
                                                   for name, module in model.named_modules() if hasattr(module, 'inference_backend')},
                             implementation_sha256=_implementation_fingerprint())

            def predict(query):
                result = engine.predict(query, beam_size=beam_size, tnorm=tnorm, return_log_scores=True)
                stats.update({key: engine.last_info[key] for key in ('raw_rows', 'cache_hits', 'pruned', 'negated_pruning')})
                return result

            def prepare(queries):
                stats.update(engine.prefetch(queries))

            def annotate(report):
                return dict(report, inference=dict(inference, statistics=dict(stats)))

            report = evaluate_benchmark(data, predict, tie_policy=tie_policy,
                                        max_queries_per_shape=max_queries_per_shape, progress=progress,
                                        prepare=prepare, query_batch_size=query_batch_size, query_order=query_order,
                                        query_sampling=query_sampling, sampling_seed=sampling_seed, on_query=on_query,
                                        checkpoint_dir=checkpoint_dir, checkpoint_identity=inference, checkpoint_every=checkpoint_every,
                                        on_checkpoint=(lambda report: on_checkpoint(annotate(report))) if on_checkpoint else None,
                                        statistics=stats)
            return annotate(report)
    finally:
        if previous_batch is not None:
            model.query_batch_size = previous_batch


def summarize_benchmarks(reports):
    """Equal dataset averages; partial selections are labeled, never padded with zeros."""
    if len({(r['dataset'], r['split']) for r in reports}) != len(reports):
        raise ValueError('Duplicate dataset/split results cannot be averaged')
    if len({r['split'] for r in reports}) > 1:
        raise ValueError('Do not aggregate validation and test splits together')
    if len({r['dataset_metadata'].get('evaluation', 'hard-answers') for r in reports}) > 1:
        raise ValueError('Do not aggregate faithfulness and hard-answer evaluations together')
    result = {}
    for group in ('all', 'transductive', 'inductive-e', 'inductive-er'):
        selected = [r for r in reports if group == 'all' or r['group'] == group]
        if not selected:
            continue
        result[group] = {'datasets': len(selected)}
        for category in ('all', 'epfo', 'negation'):
            values = [r['averages'][category] for r in selected if r['averages'][category] is not None]
            result[group][category] = ({metric: sum(v[metric] for v in values) / len(values) for metric in METRICS}
                                       | {'datasets': len(values)}) if values else None
    return dict(groups=result, complete_23_dataset_test_suite=(len(reports) == 23 and
                {r['dataset'] for r in reports} == set(BENCHMARK_DATASETS) and
                all(r['split'] == 'test' and r['protocol']['full_split'] and r['protocol']['all_14_shapes'] for r in reports)))


def add_benchmark_parser(commands):
    parser = commands.add_parser('benchmark', help='Evaluate CQD on the published UltraQuery datasets')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', default=['FB15k237LogicalQuery'], help='Official dataset names, or all for the 23-dataset suite')
    parser.add_argument('--split', choices=['valid', 'test'], default='test')
    parser.add_argument('--query-types', nargs='+', choices=list(QUERY_SHAPES))
    parser.add_argument('--download', action='store_true', help='Download official archives if missing')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--experiment', help='Saved DICE experiment')
    source.add_argument('--checkpoint', type=Path, help='Pretrained graph-model checkpoint')
    parser.add_argument('--model', choices=['ULTRA', 'TRIX', 'Flock'], default='ULTRA')
    parser.add_argument('--model-config', type=Path, help='Optional model constructor arguments as JSON')
    parser.add_argument('--adapter', type=Path, help='Fitted DICE adapter JSON')
    parser.add_argument('--observed-mix', type=float)
    parser.add_argument('--beam-size', type=int, default=64)
    parser.add_argument('--tnorm', choices=['prod', 'min'], default='prod')
    parser.add_argument('--row-batch-size', type=int, default=8)
    parser.add_argument('--backend-batch-size', type=int, help='Neural microbatch; defaults to row batch size for ULTRA/TRIX')
    parser.add_argument('--query-batch-size', type=int, default=32, help='Queries whose anchor rows are prefetched together')
    parser.add_argument('--query-order', choices=['published', 'relation'], default='relation')
    parser.add_argument('--cache-mb', type=int, default=512)
    parser.add_argument('--checkpoint-every', type=int, default=500)
    parser.add_argument('--tie-policy', choices=['sort', 'average', 'optimistic', 'pessimistic'], default='sort')
    parser.add_argument('--max-queries-per-shape', type=int, help='Deterministic smoke-test subset; omitted means full evaluation')
    parser.add_argument('--query-sampling', choices=['prefix', 'uniform'], default='prefix', help='Uniform sampling is nested across query limits')
    parser.add_argument('--sampling-seed', type=int, default=0, help='Query selection seed, independent of backbone sampling')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--precision', choices=['ieee', 'tf32'], default='ieee')
    parser.add_argument('--output', type=Path, required=True, help='JSON report with automatic checkpoint/resume')


def run_benchmark_cli(args):
    from ..models import TRIX, ULTRA, Flock
    from ..models._inference import float32_precision_backends
    from ..models.graph_model import GraphKGE
    if args.threads < 1:
        raise ValueError('Thread count must be positive')
    torch.set_num_threads(args.threads)
    backends = float32_precision_backends()
    if backends:
        for backend in backends:
            backend.fp32_precision = args.precision
    else:
        torch.set_float32_matmul_precision('highest' if args.precision == 'ieee' else 'high')
        torch.backends.cudnn.allow_tf32 = args.precision == 'tf32'
    names = list(BENCHMARK_DATASETS) if args.datasets == ['all'] else args.datasets
    if len(set(names)) != len(names):
        raise ValueError('Choose each dataset once')
    kge = None
    if args.experiment:
        from ..knowledge_graph_embeddings import KGE
        kge = KGE(path=args.experiment)
        model = kge.model
    else:
        configuration = json.loads(args.model_config.read_text()) if args.model_config else {}
        model = {'ULTRA': ULTRA, 'TRIX': TRIX, 'Flock': Flock}[args.model](dict(configuration, num_entities=1, num_relations=1))
        model.load_pretrained(args.checkpoint)
    model.to(args.device)
    adapter = QueryScoreAdapter.load(args.adapter, model=model) if args.adapter else None
    request = dict(arguments={key: str(value) for key, value in vars(args).items()},
                   implementation=_implementation_fingerprint(), backbone=state_fingerprint(model),
                   adapter=json.loads(args.adapter.read_text()) if args.adapter else None)
    directory = args.output.with_suffix(args.output.suffix + '.checkpoints')
    with BenchmarkCheckpoint(directory, request):
        try:
            _run_datasets(args, names, model, adapter, kge, directory, GraphKGE)
        except BaseException as error:
            if args.output.exists():
                payload = json.loads(args.output.read_text())
                write_json(args.output, dict(payload, state='failed', error=str(error)))
            raise


def _run_datasets(args, names, model, adapter, kge, directory, graph_type):
    reports = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(f'Loading {name} ({args.split})', flush=True)
        data = load_benchmark(args.data_root, name, split=args.split, query_types=args.query_types, download=args.download)
        if not isinstance(model, graph_type):
            if data.entity_to_idx is None or kge.entity_to_idx != data.entity_to_idx or kge.relation_to_idx != data.relation_to_idx:
                raise ValueError('Transductive experiment IDs must exactly match the published benchmark vocabulary')

        def progress(done, total):
            if done == 1 or done % 100 == 0 or done == total:
                print(f'{name}: {done}/{total} queries', flush=True)

        def checkpoint(report):
            write_json(args.output, dict(version=2, state='running', results=reports, current_result=report,
                                        summary=summarize_benchmarks(reports)))

        report = benchmark_model(model, data, adapter=adapter, observed_mix=args.observed_mix,
                                 beam_size=args.beam_size, tnorm=args.tnorm, row_batch_size=args.row_batch_size,
                                 backend_batch_size=args.backend_batch_size, query_batch_size=args.query_batch_size,
                                 query_order=args.query_order, checkpoint_dir=directory / name.replace(':', '-'),
                                 query_sampling=args.query_sampling, sampling_seed=args.sampling_seed,
                                 checkpoint_every=args.checkpoint_every, on_checkpoint=checkpoint,
                                 cache_bytes=args.cache_mb * 2**20, seed=args.seed, samples=args.samples,
                                 tie_policy=args.tie_policy, max_queries_per_shape=args.max_queries_per_shape, progress=progress)
        reports.append(report)
        write_json(args.output, dict(version=2, state='complete' if len(reports) == len(names) else 'running',
                                    results=reports, current_result=None, summary=summarize_benchmarks(reports)))
        print(f'{name}: {json.dumps(report["averages"])}', flush=True)
    print(f'Saved {args.output}', flush=True)
