"""Streaming, full-candidate evaluation of published logical queries.

The default sort/filter/query/type averaging protocol follows UltraQuery's
query_utils.py at 427966ad8ed60420eef034063d44f3153addff90. Predictions are
provided independently of labels so any query executor can use this evaluator.
"""

import hashlib
import heapq
import json
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any

import torch

from ..models._inference import to_device
from ._checkpoint import BenchmarkCheckpoint, implementation_fingerprint, write_json
from ._query import QUERY_SHAPES, ULTRAQUERY_SHAPES, compile_query, relation_signature
from ._rank_trace import RankTrace
from .context import attached_context, evaluation_mode, fingerprint, state_fingerprint
from .datasets import BENCHMARK_DATASETS, PLUS_H_DATASETS, BenchmarkQuery, QueryBenchmark, load_benchmark
from .engine import QueryAnswerer
from .score_adapter import QueryScoreAdapter

METRICS = ('mrr', 'hits1', 'hits3', 'hits10')


def _accumulate_inference_statistics(stats, info, engines):
    """Persist event counts and the combined calibrated-row cache occupancy."""
    for key in ('raw_rows', 'cache_hits', 'pruned', 'negated_pruning', 'bound_skipped', 'zero_prefix_skipped'):
        stats[key] += int(info.get(key, 0))
    stats['cache_bytes'] = sum(engine._cache_used for engine in engines)
    stats['cache_peak_bytes'] = max(stats.get('cache_peak_bytes', 0), stats['cache_bytes'])


class QueryMetrics:
    """Reuse a validated candidate domain and device masks across a dataset."""

    def __init__(self, num_entities, *, candidates=None, tie_policy='sort'):
        if tie_policy not in ('sort', 'average', 'optimistic', 'pessimistic', 'expected'):
            raise ValueError('Unknown benchmark tie policy')
        self.n, self.tie_policy = num_entities, tie_policy
        domain = range(num_entities) if candidates is None else frozenset(candidates)
        if not domain or min(domain) < 0 or max(domain) >= num_entities:
            raise ValueError('Invalid candidate domain')
        self.full_domain = len(domain) == num_entities
        self.domain = range(num_entities) if self.full_domain else domain
        self.outside = frozenset() if self.full_domain else frozenset(range(num_entities)) - domain
        self._device = None
        self._harmonic = None

    def __call__(self, scores, easy, hard):
        return self.evaluate(scores, easy, hard, (self.tie_policy,))[0][self.tie_policy]

    def evaluate(self, scores, easy, hard, policies=('sort', 'expected'), *, records=False):
        """Share one argsort across tie policies and optional answer-rank records."""
        if not policies or set(policies) - {'sort', 'expected', 'average', 'optimistic', 'pessimistic'}:
            raise ValueError('Unknown benchmark tie policy')
        if scores.ndim != 1 or len(scores) != self.n:
            raise ValueError('Expected one complete higher-is-better score vector')
        # Read with the metrics below: one device-to-host copy per evaluation.
        invalid = (torch.isnan(scores) | torch.isposinf(scores)).any()
        easy, hard = frozenset(easy), frozenset(hard)
        answers = easy | hard
        if not hard or easy & hard or any(t not in self.domain for t in answers):
            if invalid.item():
                raise ValueError('Expected one complete higher-is-better score vector')
            raise ValueError('Need disjoint easy/hard answers and nonempty hard answers inside the candidate domain')
        targets = sorted(hard)
        device = scores.device
        target_index = to_device(torch.tensor(targets), device)
        ranks = None
        if 'sort' in policies or records:
            if self._device != device:
                self._negative_template = torch.ones(self.n, dtype=torch.long, device=device)
                self._outside_index = torch.tensor(sorted(self.outside), dtype=torch.long, device=device)
                self._negative_template[self._outside_index] = 0
                self._positions = torch.arange(self.n, device=device)
                self._device = device
            # Sort before filtering, including outside-domain -inf entries, to
            # preserve the reference evaluator's tie ordering exactly.
            values = scores if scores.is_contiguous() else scores.clone()
            if not self.full_domain:
                values = scores.clone()
                values[self._outside_index] = -torch.inf
            order = values.argsort(descending=True)
            negatives = self._negative_template.clone()
            negatives.index_fill_(0, to_device(torch.tensor(sorted(answers), dtype=torch.long), device), 0)
            positions = torch.empty_like(order)
            positions[order] = self._positions
            ranks = 1 + negatives[order].cumsum(0)[positions[target_index]]
        if set(policies) != {'sort'} or records:
            excluded = sorted(self.outside | answers)
            eligible = torch.ones(self.n, dtype=torch.bool, device=device)
            eligible.index_fill_(0, to_device(torch.tensor(excluded, dtype=torch.long), device), False)
            # Eligible scores in ascending order; valid scores sort before +inf.
            count = self.n - len(excluded)
            negatives = torch.where(eligible, scores, torch.inf).sort().values[:count]
            values = scores[target_index].contiguous()
            lower = torch.searchsorted(negatives, values, right=False)
            upper = torch.searchsorted(negatives, values, right=True)
            starts, tied = count - upper + 1, upper - lower
        packed = [invalid.double()[None]]
        for policy in policies:
            if policy == 'expected':
                # Each position in the exact tie block is equally likely.
                sizes = tied + 1
                last = starts + sizes - 1
                if self._harmonic is None or self._harmonic.device != scores.device:
                    self._harmonic = torch.cat((torch.zeros(1, dtype=torch.float64, device=scores.device),
                                               torch.arange(1, self.n + 1, dtype=torch.float64, device=scores.device).reciprocal().cumsum(0)))
                rr = (self._harmonic[last] - self._harmonic[starts - 1]) / sizes
                hits = [((k - starts + 1).clamp_min(0).minimum(sizes).double() / sizes).mean() for k in (1, 3, 10)]
                metrics = torch.stack([rr.mean(), *hits])
            else:
                policy_ranks = (ranks.double() if policy == 'sort' else starts.double() +
                                {'average': .5, 'optimistic': 0., 'pessimistic': 1.}[policy] * tied)
                metrics = torch.stack([policy_ranks.reciprocal().mean(), *((policy_ranks <= k).double().mean() for k in (1, 3, 10))])
            packed.append(metrics)
        if records:
            # Integer ranks are exact in float64 for any feasible vocabulary.
            packed.append(torch.stack([target_index, ranks, starts - 1, tied + 1], dim=1).double().flatten())
        invalid, *host = torch.cat(packed).tolist()
        if invalid:
            raise ValueError('Expected one complete higher-is-better score vector')
        result = {policy: dict(zip(METRICS, host[4 * i:4 * i + 4])) for i, policy in enumerate(policies)}
        values = [int(value) for value in host[4 * len(policies):]]
        detail = [values[i:i + 4] for i in range(0, len(values), 4)] if records else None
        return result, detail


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
        digest.update(query.identity.encode())
    return digest.hexdigest()


@torch.no_grad()
def evaluate_benchmark(data: QueryBenchmark, predict: Callable[[tuple], torch.Tensor], *, tie_policy: str = 'sort',
                       max_queries_per_shape: int | None = None, progress: Callable[[int, int], None] | None = None,
                       prepare: Callable[[list[tuple]], None] | None = None, query_batch_size: int = 1,
                       query_order: str = 'published', checkpoint_dir: str | Path | None = None,
                       checkpoint_identity: Any = None, checkpoint_every: int = 500,
                       on_checkpoint: Callable[[dict[str, Any]], None] | None = None, statistics: Counter | None = None,
                       query_sampling: str = 'prefix', sampling_seed: int = 0,
                       on_query: Callable[[tuple, str, dict[str, float]], None] | None = None,
                       additional_tie_policies: Sequence[str] = (), query_plan: Sequence[BenchmarkQuery] | None = None,
                       batch_ends: Sequence[int] | None = None, rank_trace_path: str | Path | None = None,
                       comparison_predictors: Mapping[str, Callable[[tuple], torch.Tensor]] | None = None,
                       filter_corrections: Mapping[tuple, frozenset | set] | None = None) -> dict[str, Any]:
    """Stream filtered ranking metrics over a benchmark, resuming a verified query prefix.

    Each query's scores over all entities rank its hard answers after filtering
    every other easy and hard answer. Metrics average answers per query, queries
    per query type, and query types equally.

    Args:
        data: Benchmark queries, inference graph and candidates.
        predict: Scores of every entity for one query.
        tie_policy: ``'sort'`` (PyTorch's default descending argsort, as in the
            reference evaluator; not guaranteed stable, so exact ties may order
            differently across devices) or ``'expected'`` (exact expectation
            under random order within ties, which removes that dependence).
        max_queries_per_shape: Sample at most this many queries per type.
        progress: Called with completed and total query counts.
        prepare: Called with each batch of queries before they are predicted.
        query_batch_size: Queries per prepared batch.
        query_order: Order of the query plan: ``'published'``, ``'canonical'`` or ``'relation'``.
        checkpoint_dir: Directory for resumable progress.
        checkpoint_identity: Everything that determines predictions; resuming requires equality.
        checkpoint_every: Queries between saved checkpoints.
        on_checkpoint: Called with the partial report at each checkpoint.
        statistics: Counter of inference statistics to update and report.
        query_sampling: ``'prefix'`` or ``'uniform'`` sampling of limited query plans.
        sampling_seed: Seed of uniform sampling.
        on_query: Called with each query, its type and its metrics.
        additional_tie_policies: Further tie policies computed from the same scores.
        query_plan: Explicit ordered queries instead of a sampled plan.
        batch_ends: Explicit batch boundaries in ``query_plan``.
        rank_trace_path: SQLite file recording every hard answer's ranks.
        comparison_predictors: Named predictors scored on the same queries, such
            as identity-adapter controls.
        filter_corrections: Corrected easy answers by query; released filters are
            then reported as a control.

    Returns:
        The report: per-type and averaged metrics, coverage, protocol and timings.

    Raises:
        ValueError: For inconsistent settings, or a checkpoint of different inputs.
    """
    if type(query_batch_size) is not int or query_batch_size < 1 or type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError('Batch and checkpoint intervals must be positive integers')
    if query_plan is not None:
        if max_queries_per_shape is not None:
            raise ValueError('An explicit query plan cannot also specify a sampling limit')
        selected = list(query_plan)
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(data.queries):
            raise ValueError('Explicit query plan must contain distinct benchmark queries with unchanged answers')
    else:
        selected = _query_plan(data, max_queries_per_shape, query_order, sampling=query_sampling, seed=sampling_seed)
    if batch_ends is not None:
        batch_ends = list(batch_ends)
        if (not batch_ends or any(type(end) is not int for end in batch_ends)
                or batch_ends != sorted(set(batch_ends)) or batch_ends[0] < 1 or batch_ends[-1] != len(selected)):
            raise ValueError('Explicit batch boundaries must cover the entire query plan')
    if rank_trace_path is not None and checkpoint_dir is None:
        raise ValueError('Rank traces require a resumable checkpoint directory')
    comparisons: dict[str, Any] = dict(comparison_predictors or {})
    if any(not isinstance(name, str) or not name or name in ('.', '..') or Path(name).name != name
           or not callable(predictor) for name, predictor in comparisons.items()):
        raise ValueError('Comparisons require named predictors with safe directory names')
    predictors: dict[str | None, Any] = {None: predict, **comparisons}
    controls = {}
    if filter_corrections is not None:
        known = {query.query: query for query in data.queries}
        candidate_set = set(data.candidates)
        if set(filter_corrections) - known.keys():
            raise ValueError('Filter corrections reference unknown queries')
        for query, easy in filter_corrections.items():
            if set(easy) & known[query].hard or not set(easy) <= candidate_set:
                raise ValueError('Corrected filters must be valid non-target candidates')
        controls = {name: (name + '-' if name else '') + 'released-filters' for name in predictors}
        if set(controls.values()) & comparisons.keys():
            raise ValueError('Released-filter comparison names collide with predictors')
        comparisons.update(dict.fromkeys(controls.values()))
    metrics_for_query = QueryMetrics(data.context.num_entities, candidates=data.candidates, tie_policy=tie_policy)
    if len(set(additional_tie_policies)) != len(additional_tie_policies) or tie_policy in additional_tie_policies:
        raise ValueError('Tie policies must be distinct')
    additional = {policy: QueryMetrics(data.context.num_entities, candidates=data.candidates, tie_policy=policy)
                  for policy in additional_tie_policies}
    if checkpoint_dir is not None and checkpoint_identity is None:
        raise ValueError('Checkpointing requires an explicit predictor identity')
    identity = dict(predictor=checkpoint_identity, implementation=implementation_fingerprint(),
                    dataset=data.name, split=data.split, context=data.context.identity, metadata=data.metadata,
                    candidates=fingerprint(data.candidates), plan=_plan_fingerprint(selected),
                    tie_policy=tie_policy, limit=max_queries_per_shape, order=query_order,
                    additional_tie_policies=list(additional_tie_policies),
                    sampling=query_sampling, sampling_seed=sampling_seed,
                    batch=query_batch_size, checkpoint_every=checkpoint_every) if checkpoint_dir is not None else None
    if identity is not None:
        identity.update(explicit_batches=batch_ends, rank_trace=rank_trace_path is not None)
        if comparisons:
            identity['comparisons'] = list(comparisons)
        if filter_corrections is not None:
            identity['filter_corrections'] = fingerprint(sorted((fingerprint(query), sorted(easy)) for query, easy in filter_corrections.items()))
    store = BenchmarkCheckpoint(checkpoint_dir, identity) if checkpoint_dir is not None else nullcontext(None)
    available = Counter(q.shape for q in data.queries)
    stats = Counter() if statistics is None else statistics
    with ExitStack() as stack:
        checkpoint = stack.enter_context(store)
        saved = checkpoint.load() if checkpoint else None
        counts, hard_counts = Counter(), Counter()
        completed, elapsed = 0, 0.
        policies = (tie_policy, *additional)
        # Metric sums per (predictor, tie policy) and query type; predictor None is ``predict``.
        totals = {(name, policy): defaultdict(Counter) for name in (None, *comparisons) for policy in policies}
        if saved is not None:
            completed, elapsed = saved['completed'], saved['seconds']
            if not 0 <= completed <= len(selected):
                raise ValueError('Invalid checkpoint query count')
            counts.update(saved['counts'])
            hard_counts.update(saved['hard_counts'])
            for name, policy, values in saved['totals']:
                totals[name, policy].update({shape: Counter(sums) for shape, sums in values.items()})
            stats.update({key: value for key, value in saved['statistics'].items()
                          if key not in ('cache_bytes', 'cache_peak_bytes')})
            stats['cache_peak_bytes'] = max(stats.get('cache_peak_bytes', 0),
                                            saved['statistics'].get('cache_peak_bytes', 0))
            if completed == len(selected):
                # A completed checkpoint replays the last evaluation's report.
                stats['cache_bytes'] = saved['statistics'].get('cache_bytes', 0)
            else:
                # Continuing inference starts with fresh process-local caches.
                stats.setdefault('cache_bytes', 0)
            if sum(counts.values()) != completed:
                raise ValueError('Invalid checkpoint shape counts')
        if batch_ends is not None and completed and completed not in batch_ends:
            raise ValueError('Checkpoint cuts through a reference batch')
        trace, comparison_traces = None, {}
        if rank_trace_path is not None and identity is not None:
            trace = stack.enter_context(RankTrace(rank_trace_path, identity, completed))
            comparison_traces = {name: stack.enter_context(RankTrace(Path(rank_trace_path).parent / name / 'ranks.sqlite3',
                                                                     dict(identity, comparison=name), completed))
                                 for name in comparisons}
        timings = Counter(saved.get('timings', {}) if saved else {})
        started = time.monotonic()

        def report(variant: str | None = None) -> dict[str, Any]:
            def means(policy):
                return {shape: {metric: totals[variant, policy][shape][metric] / counts[shape] for metric in METRICS}
                        | dict(queries=counts[shape], hard_answers=hard_counts[shape], available_queries=available[shape])
                        for shape in sorted(counts)}

            per_shape, extra = means(tie_policy), {}
            for policy in additional:
                values = means(policy)
                extra[policy] = dict(per_shape=values, averages=_averages(values))
            result = dict(dataset=data.name, group=data.group, split=data.split, per_shape=per_shape, averages=_averages(per_shape),
                        queries=completed, seconds=elapsed + time.monotonic() - started,
                        protocol=dict(tie_policy=tie_policy, filtering='all other easy and hard answers',
                                      averaging='answers per query, queries per shape, equal shapes per group',
                                      max_queries_per_shape=max_queries_per_shape, query_order=query_order,
                                      query_sampling=query_sampling, sampling_seed=sampling_seed,
                                      full_split=completed == len(data.queries),
                                      all_14_shapes=set(counts) == set(ULTRAQUERY_SHAPES),
                                      all_benchmark_shapes=set(counts) == set(data.metadata.get('expected_query_types', ULTRAQUERY_SHAPES))),
                        dataset_metadata=data.metadata, context_sha256=data.context.identity,
                        candidate_sha256=fingerprint(data.candidates), num_candidates=len(data.candidates),
                        timings=dict(timings),
                        **({'additional_tie_metrics': extra} if extra else {}))
            if filter_corrections is not None:
                released = variant in controls.values()
                result['protocol']['answer_filter'] = 'released' if released else 'corrected'
                result['dataset_metadata'] = dict(data.metadata, answer_filter=result['protocol']['answer_filter'])
                if released:
                    result['filter_control_of'] = next(name or '' for name, control in controls.items() if control == variant)
            if variant is None and comparisons:
                result['comparisons'] = {name: report(name) for name in comparisons}
            return result

        if progress and completed:
            progress(completed, len(selected))
        while completed < len(selected):
            end = (batch_ends[bisect_right(batch_ends, completed)] if batch_ends is not None else
                   min(completed + query_batch_size, len(selected), (completed // checkpoint_every + 1) * checkpoint_every))
            batch = selected[completed:end]
            step_started = time.monotonic()
            if prepare:
                prepare([q.query for q in batch])
            timings['prepare_wall_seconds'] += time.monotonic() - step_started
            checkpoint_due = False
            for q in batch:
                for name, predictor in predictors.items():
                    step_started = time.monotonic()
                    scores = predictor(q.query)
                    timings['predict_dispatch_wall_seconds'] += time.monotonic() - step_started
                    if scores.shape != (data.context.num_entities,):
                        raise ValueError('Predictor must score the complete public entity vocabulary')
                    step_started = time.monotonic()
                    easy = q.easy if filter_corrections is None else filter_corrections.get(q.query, q.easy)
                    all_metrics, details = metrics_for_query.evaluate(scores, easy, q.hard, (tie_policy, *additional), records=trace is not None)
                    main_metrics = all_metrics[tie_policy]
                    variants = [(name, all_metrics, details)]
                    if controls:
                        released = ((all_metrics, details) if easy == q.easy else
                                    metrics_for_query.evaluate(scores, q.easy, q.hard, (tie_policy, *additional), records=trace is not None))
                        variants.append((controls[name], *released))
                    for variant, values, records in variants:
                        for policy in policies:
                            totals[variant, policy][q.shape].update(values[policy])
                        current_trace = trace if variant is None else comparison_traces.get(variant)
                        if current_trace is not None:
                            current_trace.add(completed, q, records)
                    timings['metrics_and_trace_wall_seconds'] += time.monotonic() - step_started
                    if on_query and name is None:
                        on_query(q.query, q.shape, dict(main_metrics))
                counts[q.shape] += 1
                hard_counts[q.shape] += len(q.hard)
                completed += 1
                checkpoint_due |= completed % checkpoint_every == 0 or completed == len(selected)
                if progress and completed < end:
                    progress(completed, len(selected))
            if checkpoint_due:
                current = report()
                if trace is not None:
                    trace.commit()
                    for comparison_trace in comparison_traces.values():
                        comparison_trace.commit()
                if checkpoint:
                    checkpoint.save(dict(completed=completed, seconds=current['seconds'], counts=dict(counts),
                                         hard_counts=dict(hard_counts), statistics=dict(stats), timings=dict(timings),
                                         totals=[[name, policy, dict(values)] for (name, policy), values in totals.items()]))
                if on_checkpoint:
                    on_checkpoint(current)
            if progress:
                progress(completed, len(selected))
        return report()


def benchmark_model(model: torch.nn.Module, data: QueryBenchmark, *, adapter: QueryScoreAdapter | None = None,
                    observed_mix: float | None = None, beam_size: int = 64, tnorm: str = 'prod', row_batch_size: int = 8,
                    backend_batch_size: int | None = None, cache_bytes: int = 512 * 1024 * 1024, seed: int = 0,
                    samples: int | None = None, tie_policy: str = 'sort', max_queries_per_shape: int | None = None,
                    progress: Callable[[int, int], None] | None = None, query_batch_size: int = 32,
                    query_order: str = 'relation', checkpoint_dir: str | Path | None = None, checkpoint_every: int = 500,
                    on_checkpoint: Callable[[dict[str, Any]], None] | None = None, query_sampling: str = 'prefix',
                    sampling_seed: int = 0, on_query: Callable[[tuple, str, dict[str, float]], None] | None = None,
                    executor: str = 'cqd') -> dict[str, Any]:
    """Evaluate a KGE or KGFM with CQD beam search, shared atomic caches and durable progress.

    Arguments not listed here are those of ``evaluate_benchmark``.

    Args:
        model: Link predictor scoring ``(head, relation)`` rows over all entities.
        data: Benchmark to evaluate.
        adapter: Score adapter; None uses sigmoid scores.
        observed_mix: Overrides the adapter's weight of observed facts.
        beam_size: Intermediate candidates kept per variable.
        tnorm: ``'prod'`` or ``'min'``.
        row_batch_size: Atomic rows scored per backbone call.
        backend_batch_size: Query batch of graph backbones.
        cache_bytes: Budget of the calibrated-row cache.
        seed: Seed of sampled scoring.
        samples: Number of sampled scoring passes, if any.
        executor: Query executor, ``'cqd'``.

    Returns:
        The ``evaluate_benchmark`` report with inference settings and cache statistics.
    """
    from ..models._inference import float32_precision_token
    from ..models.flock import Flock
    if executor not in ('cqd', 'qto'):
        raise ValueError('Choose cqd or qto executor')
    previous_batch = getattr(model, 'query_batch_size', None)
    effective_batch = backend_batch_size
    if effective_batch is None:
        effective_batch = previous_batch if isinstance(model, Flock) else row_batch_size
    if type(effective_batch) is not int or effective_batch < 1:
        raise ValueError('Backend batch size must be a positive integer')
    stats = Counter()
    try:
        if previous_batch is not None:
            setattr(model, 'query_batch_size', effective_batch)
        with attached_context(model, data.context), evaluation_mode(model):
            engine = QueryAnswerer(model, context=data.context, adapter=adapter, observed_mix=observed_mix,
                                   row_batch_size=row_batch_size, cache_bytes=cache_bytes, seed=seed, samples=samples)
            transform = engine.adapter
            inference = dict(method='cqd-global-prefix' if executor == 'cqd' else 'qto-exact', executor=executor,
                             model=model.name, beam_size=beam_size if executor == 'cqd' else None, tnorm=tnorm,
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
                             implementation_sha256=implementation_fingerprint())

            def predict(query):
                result = engine.predict(query, beam_size=beam_size, tnorm=tnorm, return_log_scores=True, executor=executor)
                _accumulate_inference_statistics(stats, engine.last_info, (engine,))
                return result

            def prepare(queries):
                _accumulate_inference_statistics(stats, engine.prefetch(queries), (engine,))

            def annotate(report: dict[str, Any]) -> dict[str, Any]:
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
            setattr(model, 'query_batch_size', previous_batch)


def summarize_benchmarks(reports):
    """Equal dataset averages; partial selections are labeled, never padded with zeros."""
    if len({(r['dataset'], r['split']) for r in reports}) != len(reports):
        raise ValueError('Duplicate dataset/split results cannot be averaged')
    if len({r['split'] for r in reports}) > 1:
        raise ValueError('Do not aggregate validation and test splits together')
    if len({r['dataset_metadata'].get('suite', 'ultraquery') for r in reports}) > 1:
        raise ValueError('Do not aggregate UltraQuery and +H benchmark suites together')
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
                all(r['split'] == 'test' and r['protocol']['full_split'] and r['protocol']['all_14_shapes'] for r in reports)),
                complete_plus_h_test_suite=({r['dataset'] for r in reports} == set(PLUS_H_DATASETS) and
                all(r['split'] == 'test' and r['protocol']['full_split'] and r['protocol'].get('all_benchmark_shapes', False)
                    for r in reports)))


def add_benchmark_parser(commands):
    parser = commands.add_parser('benchmark', help='Evaluate CQD/QTO on published UltraQuery or +H datasets')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', default=['FB15k237LogicalQuery'], help='Dataset names, all for UltraQuery, or +h for the three harder datasets')
    parser.add_argument('--split', choices=['valid', 'test'], default='test')
    parser.add_argument('--query-types', nargs='+', choices=list(QUERY_SHAPES))
    parser.add_argument('--download', action='store_true', help='Download official archives if missing')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--experiment', help='Saved DICE experiment')
    source.add_argument('--checkpoint', type=Path, help='Pretrained graph-model checkpoint')
    parser.add_argument('--model', choices=['ULTRA', 'TRIX', 'Flock', 'KGICL'], default='ULTRA')
    parser.add_argument('--model-config', type=Path, help='Optional model constructor arguments as JSON')
    parser.add_argument('--adapter', type=Path, help='Fitted DICE adapter JSON')
    parser.add_argument('--observed-mix', type=float)
    parser.add_argument('--beam-size', type=int, default=64)
    parser.add_argument('--executor', choices=['cqd', 'qto'], default='cqd', help='Beam search or exact QTO-style projection')
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
    from ..models import KGICL, TRIX, ULTRA, Flock
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
    names = (list(BENCHMARK_DATASETS) if args.datasets == ['all'] else
             list(PLUS_H_DATASETS) if args.datasets in (['+h'], ['+H']) else args.datasets)
    if len(set(names)) != len(names):
        raise ValueError('Choose each dataset once')
    if any(name in PLUS_H_DATASETS for name in names) and not all(name in PLUS_H_DATASETS for name in names):
        raise ValueError('Run UltraQuery and +H benchmark suites with separate output paths')
    kge = None
    if args.experiment:
        from ..knowledge_graph_embeddings import KGE
        kge = KGE(path=args.experiment)
        model = kge.model
    else:
        configuration = json.loads(args.model_config.read_text()) if args.model_config else {}
        model = {'ULTRA': ULTRA, 'TRIX': TRIX, 'Flock': Flock, 'KGICL': KGICL}[args.model](dict(configuration, num_entities=1, num_relations=1))
        model.load_pretrained(args.checkpoint)
    model.to(args.device)
    adapter = QueryScoreAdapter.load(args.adapter, model=model) if args.adapter else None
    request = dict(arguments={key: str(value) for key, value in vars(args).items()},
                   implementation=implementation_fingerprint(), backbone=state_fingerprint(model),
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
                                 beam_size=args.beam_size, executor=args.executor, tnorm=args.tnorm, row_batch_size=args.row_batch_size,
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
