"""Source-only 2i/3i adapter fitting with frozen backbone score banks.

Fits the shared inference transform using a filtered-softmax loss.
"""

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ._query import QUERY_SHAPES, combine, compile_query, nested, validate_tree
from .adapter import QueryScoreAdapter
from .context import QueryContext, attached_context, fingerprint, state_fingerprint
from .engine import AtomicScorer


@dataclass(frozen=True)
class AdapterQuery:
    query: tuple
    answers: frozenset

    def __post_init__(self):
        query = nested(self.query)
        tree = compile_query(query)
        if tree[0] != 'and' or len(tree) not in (3, 4) or any(c[0] != 'project' or c[2][0] != 'anchor' for c in tree[1:]):
            raise ValueError('Adapter training supports flat 2i/3i queries only')
        query = tuple(sorted(query))
        if len(set(query)) != len(query):
            raise ValueError('Training intersections require distinct atoms')
        object.__setattr__(self, 'query', query)
        object.__setattr__(self, 'answers', frozenset(self.answers))

    @property
    def shape(self):
        return f'{len(self.query)}i'

    @property
    def conditions(self):
        return [(h, relations[0]) for h, relations in self.query]


@dataclass(frozen=True)
class AdapterTrainingData:
    """Prepared context and complete source answer sets, with disjoint query splits."""

    name: str
    context: QueryContext
    train: tuple
    validation: tuple = ()
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.name or not self.train:
            raise ValueError('A named source and nonempty training split are required')
        seen = set()
        for item in (*self.train, *self.validation):
            tree = compile_query(item.query)
            validate_tree(tree, self.context.num_entities, self.context.num_relations)
            if item.query in seen:
                raise ValueError('Training/validation queries must be distinct, including branch permutations')
            seen.add(item.query)
            if any(type(a) is not int or not 0 <= a < self.context.num_entities for a in item.answers):
                raise ValueError('Answer outside the source vocabulary')
            easy = self.context.answers(tree)
            if not easy <= item.answers or not item.answers - easy:
                raise ValueError('Source answers must include context answers and at least one hard answer')
            if len(item.answers) == self.context.num_entities:
                raise ValueError('Training queries need at least one negative candidate')

    def to_dict(self):
        def records(items):
            return [dict(query=q.query, answers=sorted(q.answers)) for q in items]
        return dict(version=1, name=self.name, context=self.context.to_dict(), train=records(self.train),
                    validation=records(self.validation), metadata=self.metadata)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + '\n')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get('version') != 1:
            raise ValueError('Unsupported adapter data version')
        return cls(data['name'], QueryContext(**data['context']),
                   tuple(AdapterQuery(**q) for q in data['train']),
                   tuple(AdapterQuery(**q) for q in data.get('validation', [])), data.get('metadata', {}))


def prepare_adapter_data(source, *, name='source', mask_fraction=.3, train_per_shape=96,
                         validation_per_shape=32, seed=2026090851, max_attempts=100_000):
    """Mask reciprocal pairs and reuse QueryGenerator's grounding on source facts.

    ``source`` must contain source-training facts only. Never pass target test
    facts here. Answer supervision uses the source; scoring sees only the mask.
    """
    from ..query_generator import QueryGenerator
    if not 0 < mask_fraction < 1 or train_per_shape < 1 or validation_per_shape < 0 or max_attempts < 1:
        raise ValueError('Require 0 < mask_fraction < 1 and positive training/attempt counts')
    rng = random.Random(seed)
    inverse = {a: b for h, t in source.inverse_relations for a, b in ((h, t), (t, h))}
    def canonical(edge):
        h, r, t = edge
        return min(edge, (t, inverse[r], h)) if r in inverse else edge
    pairs = sorted({canonical(edge) for edge in source.triples})
    count = max(1, round(len(pairs) * mask_fraction))
    if count >= len(pairs):
        raise ValueError('Source is too small to retain a nonempty masked graph')
    hidden = set(rng.sample(pairs, count))
    context = QueryContext([edge for edge in source.triples if canonical(edge) not in hidden],
                           source.num_entities, source.num_relations, source.inverse_relations)
    generator = QueryGenerator.from_context(source, seed=seed)
    training, validation, seen = [], [], set()
    for shape in ('2i', '3i'):
        width = int(shape[0])
        targets = sorted(t for t, rs in generator.ent_in.items() if sum(map(len, rs.values())) >= width)
        examples = []
        for _ in range(max_attempts):
            if len(examples) == train_per_shape + validation_per_shape or not targets:
                break
            query = generator.tuple2list(QUERY_SHAPES[shape])
            if generator.fill_query(query, generator.ent_in, generator.ent_out, rng.choice(targets)):
                continue
            query = tuple(sorted(nested(query)))
            if query in seen:
                continue
            tree = compile_query(query)
            answers = source.answers(tree)
            if not answers - context.answers(tree) or len(answers) == source.num_entities:
                continue
            seen.add(query)
            examples.append(AdapterQuery(query, answers))
        if len(examples) != train_per_shape + validation_per_shape:
            raise ValueError(f'Only generated {len(examples)} distinct {shape} queries; reduce counts or supply prepared data')
        rng.shuffle(examples)
        training.extend(examples[:train_per_shape])
        validation.extend(examples[train_per_shape:])
    return AdapterTrainingData(name, context, tuple(training), tuple(validation),
                               dict(source_sha256=source.identity, seed=seed, mask_fraction=mask_fraction,
                                    masked_fact_pairs=count, train_per_shape=train_per_shape, validation_per_shape=validation_per_shape))


def _bank(model, data, *, cache_dir, row_batch_size, seed, samples):
    from ..models._inference import float32_precision_token
    conditions = sorted({pair for q in (*data.train, *data.validation) for pair in q.conditions})
    settings = {name: getattr(model, name, None) for name in (
        'query_batch_size', 'test_samples', 'walk_num', 'walk_len', 'refinements',
        'compact_state', 'compile_sampler', 'pack_walks', 'prefetch_walks', '_inference_backend')}
    code = hashlib.sha256()
    paths = sorted((Path(__file__).parents[1] / 'models').glob('*.py'))
    paths += [Path(__file__).with_name(name) for name in ('engine.py', 'context.py')]
    for source in paths:
        code.update(source.name.encode())
        code.update(source.read_bytes())
    identity = dict(version=1, backbone=state_fingerprint(model), context=data.context.identity,
                    conditions=conditions, seed=seed, samples=samples, row_batch_size=row_batch_size,
                    settings=settings, torch=str(torch.__version__), device=str(next(model.parameters()).device),
                    precision=float32_precision_token(), deterministic=torch.are_deterministic_algorithms_enabled(),
                    autocast=torch.is_autocast_enabled(next(model.parameters()).device.type),
                    scoring_code=code.hexdigest(), architecture=str(model),
                    configuration={k: str(v) for k, v in getattr(model, 'args', {}).items()})
    key = fingerprint(identity)
    path = Path(cache_dir) / f'{key}.pt' if cache_dir is not None else None
    if path is not None and path.exists():
        bank = torch.load(path, map_location='cpu', weights_only=True)
        if bank['identity'] != identity or bank['rows_sha256'] != state_fingerprint({'rows': bank['rows']}):
            raise ValueError('Corrupt or mismatched adapter score bank')
        raw = bank['rows']
    else:
        with attached_context(model, data.context):
            scorer = AtomicScorer(model, data.context, row_batch_size=row_batch_size, seed=seed, samples=samples)
            # Copy each scored batch off the device immediately.
            raw = torch.cat([scorer.rows(conditions[i:i + row_batch_size]).cpu() for i in range(0, len(conditions), row_batch_size)])
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(identity=identity, rows=raw, rows_sha256=state_fingerprint({'rows': raw})), path)
    if raw.shape != (len(conditions), data.context.num_entities) or not torch.isfinite(raw).all():
        raise ValueError('Invalid complete score bank')
    observed, base = data.context.features(conditions, device='cpu')
    return dict(raw=raw.to(torch.float64), observed=observed, base=base,
                indices={pair: i for i, pair in enumerate(conditions)}, identity=key)


def _query_logs(adapter, bank, query):
    indices = [bank['indices'][pair] for pair in query.conditions]
    rows = adapter(bank['raw'][indices], bank['observed'][indices], bank['base'][indices])
    return combine(list(rows), 'and', 'prod')


def filtered_softmax_loss(logs, answers, hard_answers):
    """Mean log(1 + sum_negative exp(score_negative - score_hard))."""
    negative = torch.ones_like(logs, dtype=torch.bool)
    negative[sorted(answers)] = False
    if not negative.any() or not hard_answers:
        raise ValueError('Loss requires hard answers and negative candidates')
    negative_mass = torch.logsumexp(logs[negative], 0)
    positives = logs[sorted(hard_answers)]
    return (torch.logaddexp(positives, negative_mass) - positives).mean()


def _validation(adapter, sources, banks):
    from ..evaluation._filtering import FilteredRanker
    result = {}
    ranker = FilteredRanker('optimistic')
    with torch.no_grad():
        for source, bank in zip(sources, banks):
            for shape in ('2i', '3i'):
                metrics = []
                for query in source.validation:
                    if query.shape != shape:
                        continue
                    scores = _query_logs(adapter, bank, query)
                    hard = query.answers - source.context.answers(compile_query(query.query))
                    bounds = [ranker.bounds_batch(scores[None], [answer], [sorted(query.answers)])[0] for answer in sorted(hard)]
                    ranks = [better + tied / 2 for better, tied in bounds]
                    metrics.append([sum(1 / rank for rank in ranks) / len(ranks),
                                    *(sum(rank <= k for rank in ranks) / len(ranks) for k in (1, 3, 10))])
                if metrics:
                    average = torch.tensor(metrics, dtype=torch.float64).mean(0).tolist()
                    result[f'{source.name}/{shape}'] = dict(zip(('mrr', 'hits1', 'hits3', 'hits10'), average), queries=len(metrics))
    return result


@dataclass
class AdapterFitResult:
    adapter: QueryScoreAdapter
    history: list
    validation: dict


def fit_query_adapter(model, sources, *, feature_mode=None, observed_mix=1., epochs=20, batch_size=8,
                      learning_rate=.02, identity_penalty=.001, gradient_clip=1., seed=2026090851,
                      cache_dir=None, row_batch_size=8, samples=None):
    """Fit one adapter for a frozen backbone; only 2i/3i training labels optimize it.

    Global temperature/bias is available with feature_mode='global'. Validation
    reports the fitted transform and the two unfitted sigmoid/observed baselines;
    it never selects an epoch. Caller backbone weights, graph, and modes survive.
    """
    sources = list(sources)
    if not sources or len({d.name for d in sources}) != len(sources):
        raise ValueError('Provide nonempty, uniquely named sources')
    if min(epochs, batch_size, row_batch_size) < 1 or not all(math.isfinite(v) for v in (learning_rate, identity_penalty, gradient_clip)):
        raise ValueError('Positive epoch/batch counts and finite optimizer settings required')
    if learning_rate <= 0 or identity_penalty < 0 or gradient_clip <= 0:
        raise ValueError('Invalid optimizer settings')
    feature_mode = feature_mode or ('context_scores_v1' if getattr(model, 'name', '') == 'ULTRA' else 'context')
    backbone = state_fingerprint(model)
    banks = [_bank(model, data, cache_dir=cache_dir, row_batch_size=row_batch_size, seed=seed, samples=samples) for data in sources]
    settings = dict(objective='filtered_softmax_v1', epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                    identity_penalty=identity_penalty, gradient_clip=gradient_clip, seed=seed,
                    weighting='equal_source_shape', selection='final_epoch', shapes=['2i', '3i'])
    adapter = QueryScoreAdapter(feature_mode, observed_mix, metadata=dict(
        backbone_state_sha256=backbone, training=settings, score_banks=[bank['identity'] for bank in banks],
        sources={source.name: fingerprint(source.to_dict()) for source in sources}))
    optimizer = torch.optim.Adam(adapter.parameters(), lr=learning_rate)
    examples = [(i, q) for i, source in enumerate(sources) for q in source.train]
    cells = {(i, q.shape) for i, q in examples}
    counts = {cell: sum((i, q.shape) == cell for i, q in examples) for cell in cells}
    rng, history = random.Random(seed), []
    for epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        epoch_loss = 0.
        for start in range(0, len(order), batch_size):
            optimizer.zero_grad()
            terms = []
            for index in order[start:start + batch_size]:
                i, query = examples[index]
                hard = query.answers - sources[i].context.answers(compile_query(query.query))
                logs = _query_logs(adapter, banks[i], query)
                weight = len(examples) / (len(cells) * counts[i, query.shape])
                terms.append(weight * filtered_softmax_loss(logs, query.answers, hard))
            loss = torch.stack(terms).sum() / batch_size + identity_penalty * adapter.weights.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite adapter training objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            epoch_loss += float(loss.detach())
        history.append(dict(epoch=epoch + 1, loss=epoch_loss / math.ceil(len(examples) / batch_size)))
    if state_fingerprint(model) != backbone:
        raise RuntimeError('Backbone changed during adapter fitting')
    variants = {'sigmoid': QueryScoreAdapter('global'), 'observed': QueryScoreAdapter('global', 1.), 'fitted': adapter}
    validation = {name: _validation(value, sources, banks) for name, value in variants.items()}
    return AdapterFitResult(adapter, history, validation)
