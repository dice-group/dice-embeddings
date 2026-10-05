"""Shared evaluation of checkpoint-compatible CQA methods and KGFM adapters."""

import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, cast

import torch

from ._checkpoint import checksum, write_json
from ._query import QUERY_SHAPES, compile_query
from .benchmark import _accumulate_inference_statistics, _query_plan, evaluate_benchmark
from .catalog import GRAPH_INDEPENDENT_METHODS, METHODS, check_recipe, query_types_for_dataset  # noqa: F401
from .context import attached_context, evaluation_mode, fingerprint
from .datasets import QueryBenchmark, load_benchmark, source_query_groups
from .engine import AtomicBatchCache, QueryAnswerer
from .methods import CLMPT, REFERENCES, ConE, load_method
from .methods._common import FuzzyGNN
from .score_adapter import QueryScoreAdapter


def save_results(report: dict[str, Any], output: str | Path, *, run: dict[str, Any], metadata: dict[str, Any],
                 rank_trace_path: str | Path | None = None) -> dict[str, Any]:
    """Finalize shared provenance, paired results, and rank checksums together."""
    output = Path(output)
    trace = Path(rank_trace_path) if rank_trace_path is not None else output / 'ranks.sqlite3'
    for name, comparison in report.pop('comparisons', {}).items():
        filter_control = comparison.pop('filter_control_of', None)
        if filter_control is None:
            comparison['paired_with'] = run['entry']
        else:
            comparison['filter_paired_with'] = run['entry'] + ('-' + filter_control if filter_control else '')
            if filter_control:
                comparison['paired_with'] = run['entry'] + '-released-filters'
        comparison.update(metadata, benchmark_run=dict(run, entry=f'{run["entry"]}-{name}'),
                          rank_trace_sha256=checksum(trace.parent / name / 'ranks.sqlite3'))
        write_json(output / name / 'result.json', comparison)
        report.setdefault('comparison_results', {})[name] = checksum(output / name / 'result.json')
    report.update(metadata, benchmark_run=run, rank_trace_sha256=checksum(trace))
    write_json(output / 'result.json', report)
    return report


def evaluate_method(entry: Mapping[str, Any], *, output: str | Path, device: str = 'cpu', split: str = 'test',
                    limit: int | None = None, seed: int = 0, threads: int | None = None, data: QueryBenchmark | None = None,
                    query_plan: list | None = None, batch_ends: list[int] | None = None,
                    rank_trace_path: str | Path | None = None, provenance: dict | None = None,
                    query_order: str = 'relation', write_result: bool = True,
                    on_prediction: Callable[[tuple, Any], None] | None = None, reference_batches: dict | None = None,
                    on_control_prediction: Callable[[tuple, Any], None] | None = None,
                    filter_corrections: Mapping[tuple, frozenset | set] | None = None) -> dict[str, Any]:
    """Evaluate a published method or a KGFM adapter recipe with shared metrics and resumable progress.

    Args:
        entry: Recipe with ``method``, ``checkpoint``, ``dataset``, ``root`` and
            optional ``options``, ``query_types``, ``query_batch_size`` and
            ``inference_graph``. Transductive embedding methods also bind
            ``vocabulary_dataset``; KGFM adapters specify ``adapters``,
            per-type ``operators`` and ``selection_protocol``.
        output: Directory for ``result.json``, rank traces and progress.
        device: Inference device.
        split: ``'valid'`` or ``'test'``.
        limit: Uniform sample of at most this many queries per type.
        seed: Sampling seed.
        threads: CPU threads for PyTorch.
        data: Preloaded benchmark matching ``entry`` and ``split``.
        query_plan: Explicit ordered queries.
        batch_ends: Explicit batch boundaries in ``query_plan``.
        rank_trace_path: Rank trace file (default: ``output/ranks.sqlite3``).
        provenance: Identity of a frozen study, replacing the direct-run identity.
        query_order: ``'relation'``, ``'canonical'`` or ``'upstream-pickle'``.
        write_result: Write ``result.json`` with run metadata.
        on_prediction: Called with each query and its scores.
        reference_batches: Upstream batch of each query's first member.
        on_control_prediction: Called for identity-adapter control predictions.
        filter_corrections: Corrected easy answers by query.

    Returns:
        The evaluation report.

    Raises:
        ValueError: If the recipe or supplied data are invalid.
    """
    if entry.get('method') not in METHODS:
        raise ValueError(f'Choose a supported query method: {METHODS}')
    if data is not None and (data.name != entry.get('dataset') or data.split != split):
        raise ValueError('Loaded benchmark does not match the requested dataset and split')
    if data is not None and entry.get('inference_graph') is not None:
        expected_graph = entry['inference_graph'] if split == 'test' else 'train'
        if data.metadata.get('inference_graph') != expected_graph:
            raise ValueError('Loaded benchmark does not match the requested inference graph')
    kgfm = entry['method'] not in REFERENCES
    required = {'method', 'checkpoint', 'dataset', 'root', *(('adapters', 'operators', 'selection_protocol') if kgfm else ())}
    allowed = required | {'options', 'query_types', 'query_batch_size', 'id', 'inference_graph',
                          *(('adapter_ablation',) if kgfm else ('vocabulary_dataset', 'per_shape'))}
    if required - entry.keys() or entry.keys() - allowed:
        raise ValueError(f'Expected fields {sorted(required)}; unexpected fields {sorted(entry.keys() - allowed)}')
    types = entry.get('query_types')
    if types is not None and (not types or set(types) - QUERY_SHAPES.keys() or len(types) != len(set(types))):
        raise ValueError('Invalid query type selection')
    check_recipe(dict(entry), types or query_types_for_dataset(entry['dataset']))
    output = Path(output)
    rank_trace_path = Path(rank_trace_path) if rank_trace_path is not None else output / 'ranks.sqlite3'
    if query_plan is None and query_order == 'canonical':
        query_order = 'published'
    if query_plan is None and query_order == 'upstream-pickle':
        data = data if data is not None else load_benchmark(entry['root'], entry['dataset'], split=split,
                                                           query_types=entry.get('query_types'), inference_graph=entry.get('inference_graph'))
        selected = set(_query_plan(data, limit, 'published', sampling='uniform', seed=seed))
        shapes = entry.get('query_types') or list(dict.fromkeys(query.shape for query in data.queries))
        size = entry.get('query_batch_size', 1)
        query_plan, batch_ends, reference_batches = [], [], dict(reference_batches or {})
        for group in source_query_groups(data, entry['root'], shapes):
            for start in range(0, len(group), size):
                batch = group[start:start + size]
                probes = [query for query in batch if query in selected]
                if probes:
                    reference_batches[probes[0].query] = [query.query for query in batch]
                    query_plan.extend(probes)
                    batch_ends.append(len(query_plan))
        limit = None
    direct_run = dict(entry=entry.get('id', f'{entry["method"]}-{entry["dataset"]}'),
                      phase='test' if split == 'test' else 'pilot', mode='direct')
    direct_metadata = {'reference': {'status': 'direct evaluation; no frozen-study verification'}}
    if entry['method'] in REFERENCES:
        if on_control_prediction is not None:
            raise ValueError('Identity calibration controls require a KGFM adapter method')
        report = _evaluate_checkpoint_method(
            entry, output=output, device=device, split=split, limit=limit, seed=seed, threads=threads,
            data=data, query_plan=query_plan, batch_ends=batch_ends, rank_trace_path=rank_trace_path,
            provenance=provenance, query_order=query_order,
            on_prediction=on_prediction, reference_batches=reference_batches, filter_corrections=filter_corrections)
        return save_results(report, output, run=direct_run, metadata=direct_metadata,
                            rank_trace_path=rank_trace_path) if write_result else report
    options = entry.get('options', {})
    if threads is not None:
        if threads < 1:
            raise ValueError('Thread count must be positive')
        torch.set_num_threads(threads)
    from ..models._inference import float32_precision_backends
    for backend in float32_precision_backends():
        backend.fp32_precision = 'ieee'
    started = time.monotonic()
    data = data if data is not None else load_benchmark(entry['root'], entry['dataset'], split=split,
                                                       query_types=entry.get('query_types'), inference_graph=entry.get('inference_graph'))
    identity = provenance if provenance is not None else dict(
        manifest=entry, checkpoint_sha256=checksum(entry['checkpoint']),
        adapters={name: checksum(path) for name, path in entry['adapters'].items()},
        device=str(device), torch=str(torch.__version__), threads=torch.get_num_threads(), seed=seed)
    report: dict[str, Any] = _evaluate_kgfm(
        dict(entry, options=options), data, query_plan, batch_ends, output, device, identity, on_prediction, reference_batches,
        on_control_prediction, limit=limit, seed=seed, query_order=query_order, rank_trace_path=rank_trace_path,
        filter_corrections=filter_corrections)
    report['total_wall_seconds'] = time.monotonic() - started
    report['manifest_sha256'] = fingerprint(entry)
    if write_result:
        save_results(report, output, run=direct_run, metadata=direct_metadata, rank_trace_path=rank_trace_path)
    return report


def _evaluate_checkpoint_method(entry, *, output, device='cpu', split='test', limit=None, seed=0, threads=None,
              data=None, query_plan=None, batch_ends=None, rank_trace_path=None, provenance=None, query_order='relation',
              on_prediction=None, reference_batches=None, filter_corrections=None):
    """One dataset/model at a time; both tie protocols share every prediction."""
    method, dataset, types = entry['method'], entry['dataset'], entry.get('query_types')
    if method not in ('ultraquery', 'ultraquery-lp', 'incoming-relation') and entry.get('vocabulary_dataset') != dataset:
        raise ValueError('Bind transductive checkpoint IDs explicitly with vocabulary_dataset equal to dataset')
    if threads is not None:
        if threads < 1:
            raise ValueError('Thread count must be positive')
        torch.set_num_threads(threads)
    torch.set_float32_matmul_precision('highest')
    load_started = time.monotonic()
    data = data if data is not None else load_benchmark(entry['root'], dataset, split=split, query_types=types,
                                                       inference_graph=entry.get('inference_graph'))
    options = dict(entry.get('options', {}))
    if method == 'incoming-relation':
        options.setdefault('seed', seed)
    model = load_method(method, entry['checkpoint'], data.context, device=device, **options)
    shape_by_query = {item.query: item.shape for item in data.queries}
    overrides = entry.get('per_shape', {})
    if set(overrides) - set(shape_by_query.values()):
        raise ValueError('Per-shape configuration names must occur in the selected benchmark')
    permitted = {'beam_size', 'tnorm', 'max_k'} if method in ('cqd', 'cqd-hybrid') else set()
    if any(set(value) - permitted for value in overrides.values()):
        raise ValueError('Only CQD beam_size, tnorm and max_k permit per-shape overrides')
    defaults = {key: getattr(model, key) for key in permitted}
    for value in overrides.values():
        if ('tnorm' in value and value['tnorm'] not in ('prod', 'min') or
                any(key in value and (type(value[key]) is not int or value[key] < 1) for key in ('beam_size', 'max_k'))):
            raise ValueError('Invalid per-shape CQD configuration')
    batch_size = entry.get('query_batch_size', 1)
    prepared = {}

    def prepare(queries):
        prepared.clear()
        if isinstance(model, (CLMPT, ConE, FuzzyGNN)):
            requested = set(queries)
            queries = (reference_batches or {}).get(queries[0], queries)
            groups = defaultdict(list)
            for query in queries:
                groups[shape_by_query[query]].append(query)
            for group in groups.values():
                if isinstance(model, FuzzyGNN):
                    scores = model.predict_batch(group)
                    prepared.update({query: scores[i] for i, query in enumerate(group) if query in requested})
                else:
                    branches = model.encode_trees([compile_query(query) for query in group])
                    prepared.update({query: [branch[i:i + 1] for branch in branches]
                                     for i, query in enumerate(group) if query in requested})

    def predict(query):
        if query in prepared:
            value = prepared.pop(query)
            scores = value if isinstance(model, FuzzyGNN) else cast(CLMPT | ConE, model).score_embeddings(value)[0]
        else:
            for key, value in (defaults | overrides.get(shape_by_query[query], {})).items():
                setattr(model, key, value)
            scores = model.predict(query)
        if on_prediction:
            on_prediction(query, scores)
        return scores

    identity = dict(**model.provenance, manifest=entry, torch=torch.__version__, device=device,
                    cuda=torch.version.cuda, threads=torch.get_num_threads(), matmul_precision='highest',
                    batch_size=batch_size)
    if provenance is not None:
        identity['paper_protocol'] = provenance
    load_seconds = time.monotonic() - load_started
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    report = evaluate_benchmark(data, predict, tie_policy='sort', additional_tie_policies=('expected',),
                                max_queries_per_shape=limit, query_sampling='uniform', sampling_seed=seed,
                                query_order=query_order, query_batch_size=batch_size, prepare=prepare,
                                query_plan=query_plan, batch_ends=batch_ends, rank_trace_path=rank_trace_path,
                                checkpoint_dir=destination / 'progress', checkpoint_identity=identity,
                                filter_corrections=filter_corrections,
                                checkpoint_every=100, on_checkpoint=lambda report: write_json(destination / 'partial.json', report))
    report['inference'] = identity
    for comparison in report.get('comparisons', {}).values():
        comparison['inference'] = identity
    report['load_seconds'] = load_seconds
    report['manifest_sha256'] = fingerprint(entry)
    return report


@contextmanager
def deterministic_kgfm():
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def _evaluate_kgfm(entry, data, plan, ends, output, device, identity, on_prediction=None, reference_batches=None,
                   on_control_prediction=None, *, limit=None, seed=0, query_order='relation', rank_trace_path=None,
                   filter_corrections=None):
    from ..models import KGICL, TRIX, ULTRA
    backbone = entry['method'].split('-')[0]
    options = entry['options']
    allowed = {'beam_size', 'row_batch_size', 'backend_batch_size', 'cache_bytes', 'raw_cache_bytes', 'observed_facts',
               'raw_cache_device', 'raw_cache_granularity', 'relation_cache_mb', 'projection_cache_mb'}
    if backbone == 'kgicl':
        allowed |= {'prompt_seed'}
    if options.keys() - allowed:
        raise ValueError('Unknown KGFM execution option')
    if options.get('raw_cache_device', 'cpu') not in ('cpu', 'model'):
        raise ValueError('Atomic cache device must be cpu or model')
    granularity = options.get('raw_cache_granularity', 'batch')
    if granularity not in ('batch', 'row'):
        raise ValueError('Atomic cache granularity must be batch or row')
    # KG-ICL samples prompt graphs from the attached context with a fixed seed (default 0);
    # options without a default here keep each backbone's construction unchanged.
    extra = {'kgicl_prompt_seed': options['prompt_seed']} if 'prompt_seed' in options else {}
    model = {'ultra': ULTRA, 'trix': TRIX, 'kgicl': KGICL}[backbone](dict(
        num_entities=1, num_relations=1, graph_inference_backend='auto',
        graph_relation_cache_mb=options.get('relation_cache_mb', 64),
        graph_projection_cache_mb=options.get('projection_cache_mb', 64),
        **{f'{backbone}_query_batch_size': options.get('backend_batch_size', 1)}, **extra)).load_pretrained(
            entry['checkpoint']).to(device).eval().requires_grad_(False)
    adapters = {name: QueryScoreAdapter.load(path, model=model) for name, path in entry['adapters'].items()}
    for operator, adapter in adapters.items():
        expected = 'prod' if operator.partition(':')[0] == 'product' else 'min'
        if adapter.metadata.get('training', {}).get('tnorm', 'prod') != expected:
            raise ValueError('Operator must use an adapter fitted with the matching t-norm')
    observed_facts = options.get('observed_facts')
    if observed_facts is not None:
        for adapter in adapters.values():
            adapter.observed_mix = float(observed_facts != 'none')
    restore_observed = observed_facts in (None, 'both')
    shape_by_query = {query.query: query.shape for query in data.queries}
    stats = Counter()
    paired = entry.get('adapter_ablation', False)
    # Row granularity also helps a single configuration: beams of different queries share rows.
    raw_cache = AtomicBatchCache(options.get('raw_cache_bytes', 2 * 1024**3), device=options.get('raw_cache_device', 'cpu'),
                                 granularity=granularity) if paired or granularity == 'row' else None
    with ExitStack() as stack:
        stack.enter_context(deterministic_kgfm())
        stack.enter_context(attached_context(model, data.context))
        stack.enter_context(evaluation_mode(model))
        recipes = {'learned': adapters}
        if paired:
            recipes['without-adapter'] = {name: QueryScoreAdapter('global', adapter.observed_mix,
                normalization=adapter.normalization, metadata={'calibration': 'identity'}) for name, adapter in adapters.items()}
        engines = {variant: {name: QueryAnswerer(model, context=data.context, adapter=adapter,
                                      restore_observed=restore_observed,
                                      row_batch_size=options.get('row_batch_size', 1),
                                      raw_cache=raw_cache,
                                      cache_bytes=options.get('cache_bytes', 512 * 1024**2) // (len(adapters) * len(recipes)))
                             for name, adapter in recipe.items()} for variant, recipe in recipes.items()}
        all_engines = [engine for group in engines.values() for engine in group.values()]

        def record(info):
            _accumulate_inference_statistics(stats, info, all_engines)
            if raw_cache is not None:
                stats['backbone_computed_rows'] = initial_raw['computed'] + raw_cache.computed
                stats['backbone_reused_rows'] = initial_raw['reused'] + raw_cache.hits
                stats['raw_cache_peak_bytes'] = max(stats.get('raw_cache_peak_bytes', 0), raw_cache.peak)

        initial_raw = {}

        def initialize_raw_counts():
            if not initial_raw:
                initial_raw.update(computed=stats.get('backbone_computed_rows', 0), reused=stats.get('backbone_reused_rows', 0))

        def prepare(queries):
            initialize_raw_counts()
            queries = (reference_batches or {}).get(queries[0], queries)
            for group in engines.values():
                for operator, engine in group.items():
                    selected = [query for query in queries if entry['operators'][shape_by_query[query]] == operator]
                    if selected:
                        record(engine.prefetch(selected))

        def predict(query, variant='learned'):
            operator = entry['operators'][shape_by_query[query]]
            engine = engines[variant][operator]
            result = engine.predict(query, beam_size=options.get('beam_size', 64),
                                    tnorm='prod' if operator.partition(':')[0] == 'product' else 'min', return_log_scores=True)
            record(engine.last_info)
            callback = on_prediction if variant == 'learned' else on_control_prediction
            if callback:
                callback(query, result)
            return result

        report = evaluate_benchmark(data, predict, additional_tie_policies=('expected',), query_plan=plan,
                                    prepare=prepare, statistics=stats, query_order=query_order,
                                    query_batch_size=entry.get('query_batch_size', 1), max_queries_per_shape=limit,
                                    query_sampling='uniform', sampling_seed=seed,
                                    batch_ends=ends, checkpoint_dir=output / 'progress', checkpoint_identity=identity,
                                    checkpoint_every=100, rank_trace_path=rank_trace_path,
                                    filter_corrections=filter_corrections,
                                    on_checkpoint=lambda report: write_json(output / 'partial.json', report),
                                    comparison_predictors={'without-adapter': lambda query: predict(query, 'without-adapter')} if paired else None)
    for variant, result in [('learned', report), *report.get('comparisons', {}).items()]:
        calibration = result.get('filter_control_of', variant) or 'learned'
        result['inference'] = dict(method=entry['method'], options=options, operators=entry['operators'], deterministic_algorithms=True,
                                   calibration=calibration, selection_protocol=entry['selection_protocol'], cache_statistics=dict(stats),
                                   observed_facts=dict(mode=observed_facts or 'checkpoint-default',
                                       atomic_mix={name: adapter.observed_mix for name, adapter in adapters.items()},
                                       restore_answers=restore_observed),
                                   runtime_scope='paired adapter evaluation' if paired else 'single configuration')
    return report
