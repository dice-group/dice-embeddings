"""Source-only adapter fitting with frozen backbone score banks.

Fits the shared inference transform using a filtered-softmax loss.
"""

import copy
import hashlib
import json
import math
import os
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ._query import QUERY_SHAPES, ULTRAQUERY_SHAPES, atomic_conditions, compile_query, execute_query, nested, positive, validate_tree
from .adapter import QueryScoreAdapter
from .context import QueryContext, attached_context, fingerprint, state_fingerprint
from .engine import AtomicScorer

FLAT_SHAPES = ('2i', '3i', '2in', '3in')
TRAINING_SHAPES = ULTRAQUERY_SHAPES
STANDARD_TRAINING_SHAPES = ('1p', '2p', '3p', '2i', '3i', '2in', '3in', 'inp', 'pin', 'pni')


def _canonical_query(query):
    if isinstance(query, (int, str)):
        return query
    if len(query) == 2 and all(isinstance(v, int) or v in ('r', 'n') for v in query[1]) and query[1] not in ((-1,), ('u',)):
        return _canonical_query(query[0]), tuple(query[1])
    union = query[-1] in ((-1,), ('u',))
    branches = [_canonical_query(q) for q in (query[:-1] if union else query)]
    if all(len(q) == 2 and isinstance(q[0], int) for q in branches):
        branches.sort(key=lambda q: (len(q[1]), q))
    else:
        branches.sort(key=repr)
    return (*branches, query[-1]) if union else tuple(branches)


def _structure(query):
    if isinstance(query, int):
        return 'e'
    if len(query) == 2 and all(isinstance(v, int) for v in query[1]) and query[1] != (-1,):
        return _structure(query[0]), tuple('n' if r == -2 else 'r' for r in query[1])
    union = query[-1] == (-1,)
    branches = tuple(sorted((_structure(q) for q in (query[:-1] if union else query)), key=repr))
    return (*branches, ('u',)) if union else branches


_SHAPE_NAMES = {_canonical_query(QUERY_SHAPES[s]): s for s in TRAINING_SHAPES}


@dataclass(frozen=True)
class AdapterQuery:
    query: tuple
    answers: frozenset

    def __post_init__(self):
        query = _canonical_query(nested(self.query))
        tree = compile_query(query)
        shape = _SHAPE_NAMES.get(_structure(query))
        if shape is None:
            raise ValueError('Unsupported adapter training query structure')
        conditions = list(atomic_conditions(tree))
        if shape in FLAT_SHAPES and len(set(conditions)) != len(conditions):
            raise ValueError('Training intersections require distinct atoms')
        object.__setattr__(self, 'query', query)
        object.__setattr__(self, 'answers', frozenset(self.answers))

    @property
    def shape(self):
        return _SHAPE_NAMES[_structure(self.query)]

    @property
    def conditions(self):
        return list(atomic_conditions(compile_query(self.query)))


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
            if ('n' not in item.shape and not easy <= item.answers) or not item.answers - easy:
                raise ValueError('Source answers must contain hard answers and all positive context proofs')
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
                         validation_per_shape=32, seed=2026090851, max_attempts=100_000, shapes=('2i', '3i'),
                         extend=None, train_counts=None):
    """Mask reciprocal pairs and reuse QueryGenerator's grounding on source facts.

    ``source`` must contain source-training facts only. Never pass target test
    facts here. Answer supervision uses the source; scoring sees only the mask.
    """
    from ..query_generator import QueryGenerator
    if not 0 < mask_fraction < 1 or train_per_shape < 1 or validation_per_shape < 0 or max_attempts < 1:
        raise ValueError('Require 0 < mask_fraction < 1 and positive training/attempt counts')
    shapes = tuple(shapes)
    if not shapes or len(set(shapes)) != len(shapes) or any(s not in TRAINING_SHAPES for s in shapes):
        raise ValueError(f'Choose distinct training shapes from {TRAINING_SHAPES}')
    counts = {s: train_per_shape for s in shapes} if train_counts is None else dict(train_counts)
    if set(counts) != set(shapes) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError('Positive training counts required for every prepared shape')
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
    if extend is not None:
        if (extend.name != name or extend.context.identity != context.identity
                or extend.metadata.get('source_sha256') != source.identity
                or set(q.shape for q in (*extend.train, *extend.validation)) - set(shapes)):
            raise ValueError('Extension requires the same source, mask, name, and shapes')
        for shape in shapes:
            if (sum(q.shape == shape for q in extend.train) > counts[shape]
                    or sum(q.shape == shape for q in extend.validation) != validation_per_shape):
                raise ValueError('Extension must retain all training and validation queries')
    generator = QueryGenerator.from_context(source, seed=seed)
    training = list(extend.train) if extend is not None else []
    validation = list(extend.validation) if extend is not None else []
    seen = {q.query for q in (*training, *validation)}
    for shape in shapes:
        width = int(shape[0]) if shape in FLAT_SHAPES else 1
        targets = sorted(t for t, rs in generator.ent_in.items() if sum(map(len, rs.values())) >= width)
        examples = []
        required = (counts[shape] - sum(q.shape == shape for q in training) if extend is not None
                    else counts[shape] + validation_per_shape)
        for _ in range(max_attempts):
            if len(examples) == required or not targets:
                break
            query = generator.tuple2list(QUERY_SHAPES[shape])
            if generator.fill_query(query, generator.ent_in, generator.ent_out, rng.choice(targets)):
                continue
            query = _canonical_query(nested(query))
            if query in seen:
                continue
            tree = compile_query(query)
            answers = source.answers(tree)
            context_answers = context.answers(tree)
            if not answers - context_answers or len(answers) == source.num_entities:
                continue
            if 'n' in shape and not context_answers - answers:
                continue
            seen.add(query)
            examples.append(AdapterQuery(query, answers))
        if len(examples) != required:
            raise ValueError(f'Only generated {len(examples)} distinct {shape} queries; reduce counts or supply prepared data')
        rng.shuffle(examples)
        if extend is None:
            training.extend(examples[:counts[shape]])
            validation.extend(examples[counts[shape]:])
        else:
            training.extend(examples)
    return AdapterTrainingData(name, context, tuple(training), tuple(validation),
                               dict(extend.metadata if extend is not None else {},
                                    source_sha256=source.identity, seed=seed, mask_fraction=mask_fraction,
                                    masked_fact_pairs=count, train_per_shape=train_per_shape,
                                    train_counts=counts,
                                    validation_per_shape=validation_per_shape, shapes=list(shapes)))


def _bank(model, data, *, cache_dir, row_batch_size, seed, samples, device='cpu',
          cache_bytes=128 * 1024**2, device_cache_bytes=0):
    if any(q.shape not in FLAT_SHAPES for q in (*data.train, *data.validation)):
        from ._training_rows import TrainingRows
        provider = TrainingRows(model, data.context, cache_dir=cache_dir, row_batch_size=row_batch_size,
                                seed=seed, samples=samples, device=device,
                                cache_bytes=cache_bytes, device_cache_bytes=device_cache_bytes)
        return dict(provider=provider, identity=provider.identity, context=data.context)
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
                indices={pair: i for i, pair in enumerate(conditions)}, identity=key, context=data.context)


def _prepared_bank(adapter, bank):
    if 'provider' in bank:
        return bank
    device = adapter.weights.device
    if bank.get('prepared_configuration') == adapter.configuration and bank.get('prepared_device') == str(device):
        return bank
    raw, observed, features = [], [], []
    for start in range(0, len(bank['raw']), 32):
        end = start + 32
        r, o, f = adapter.prepare(bank['raw'][start:end], bank['observed'][start:end], bank['base'][start:end])
        raw.append(r)
        observed.append(o)
        features.append(f)
    return dict(bank, prepared_raw=torch.cat(raw).to(device), prepared_observed=torch.cat(observed).to(device),
                features=torch.cat(features).to(device),
                prepared_configuration=adapter.configuration, prepared_device=str(device))


def _context_answers(bank, query):
    cache = bank.setdefault('context_answers', {})
    if query.query not in cache:
        cache[query.query] = bank['context'].answers(compile_query(query.query))
    return cache[query.query]


def _query_logs(adapter, bank, query):
    context = bank['context']
    provider = bank.get('provider')
    batch_size = 32 if provider is not None else 8

    def rows(heads, relation):
        for start in range(0, len(heads), batch_size):
            batch = heads[start:start + batch_size]
            conditions = [(head, relation) for head in batch]
            if provider is not None:
                raw, observed, features = provider.prepared(adapter, conditions, device=adapter.weights.device)
            else:
                indices = [bank['indices'][pair] for pair in conditions]
                raw, observed, features = (bank[k][indices] for k in ('prepared_raw', 'prepared_observed', 'features'))
            yield batch, adapter.transform(raw, observed, features)

    trees = bank.setdefault('trees', {})
    if query.query not in trees:
        trees[query.query] = compile_query(query.query)
    tree = trees[query.query]
    result, _ = execute_query(tree, rows, n=context.num_entities, device=adapter.weights.device, row_batch_size=batch_size,
                              beam_size=bank.get('beam_size', 64), tnorm=bank.get('tnorm', 'prod'))
    if adapter.observed_mix == 1 and positive(tree):
        result = result.clone()
        result[sorted(_context_answers(bank, query))] = 0.
    return result


def filtered_softmax_loss(logs, answers, hard_answers):
    """Mean log(1 + sum_negative exp(score_negative - score_hard))."""
    negative = torch.ones_like(logs, dtype=torch.bool)
    negative[sorted(answers)] = False
    if not negative.any() or not hard_answers:
        raise ValueError('Loss requires hard answers and negative candidates')
    positives = logs[sorted(hard_answers)]
    negative_scores = logs[negative]
    if torch.isneginf(negative_scores).all():
        # Exact negation can rule out every negative. The loss is then zero;
        # logsumexp of all -inf values would introduce undefined gradients.
        return positives.sum() * 0.
    negative_mass = torch.logsumexp(negative_scores, 0)
    return (torch.logaddexp(positives, negative_mass) - positives).mean()


def _filtered_ranks(scores, answers, hard):
    eligible = torch.ones_like(scores, dtype=torch.bool)
    eligible[sorted(answers)] = False
    negatives, targets = scores[eligible], scores[sorted(hard)]
    if targets.isnan().any() or negatives.isnan().any():
        raise ValueError('Cannot rank NaN prediction scores')
    if len(hard) < 8:
        better = (negatives[None] > targets[:, None]).sum(1)
        tied = (negatives[None] == targets[:, None]).sum(1)
    else:
        ordered = negatives.sort().values
        left = torch.searchsorted(ordered, targets)
        right = torch.searchsorted(ordered, targets, right=True)
        better, tied = len(ordered) - right, right - left
    return (1 + better + tied.to(torch.float64) / 2).tolist()


def _validation(adapter, sources, banks):
    result = {}
    device = banks[0].get('validation_device', adapter.weights.device) if banks else adapter.weights.device
    if torch.device(device) != adapter.weights.device:
        adapter = copy.deepcopy(adapter).to(device)
    with torch.no_grad():
        for source, bank in zip(sources, banks):
            bank = _prepared_bank(adapter, bank)
            for shape in bank.get('validation_shapes', TRAINING_SHAPES):
                metrics = []
                for query in source.validation:
                    if query.shape != shape:
                        continue
                    scores = _query_logs(adapter, bank, query)
                    hard = query.answers - _context_answers(bank, query)
                    ranks = _filtered_ranks(scores, query.answers, hard)
                    metrics.append([sum(1 / rank for rank in ranks) / len(ranks),
                                    *(sum(rank <= k for rank in ranks) / len(ranks) for k in (1, 3, 10))])
                if metrics:
                    average = torch.tensor(metrics, dtype=torch.float64).mean(0).tolist()
                    result[f'{source.name}/{shape}'] = dict(zip(('mrr', 'hits1', 'hits3', 'hits10'), average), queries=len(metrics))
    return result


def _baseline_validation(adapter, sources, banks, cache_dir):
    from ._checkpoint import write_json
    from .benchmark import _implementation_fingerprint
    if cache_dir is None:
        return _validation(adapter, sources, banks)
    identity = fingerprint(dict(adapter=adapter.to_dict(), implementation=_implementation_fingerprint(),
                                cpu_threads=torch.get_num_threads(),
                                sources=[fingerprint(s.to_dict()) for s in sources],
                                banks=[b['identity'] for b in banks],
                                execution=[dict(beam=b.get('beam_size', 64), tnorm=b.get('tnorm', 'prod'), shapes=b.get('validation_shapes', TRAINING_SHAPES),
                                                device=str(b.get('validation_device', b.get('device', 'cpu')))) for b in banks]))
    path = Path(cache_dir)/f'validation-{identity}.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['identity'] != identity or saved['sha256'] != fingerprint(saved['metrics']):
            raise ValueError('Corrupt baseline validation cache')
        return saved['metrics']
    metrics = _validation(adapter, sources, banks)
    write_json(path, dict(identity=identity, metrics=metrics, sha256=fingerprint(metrics)))
    return metrics


@dataclass
class AdapterFitResult:
    adapter: QueryScoreAdapter
    history: list
    validation: dict


def _save_fit(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def fit_query_adapter(model, sources, *, feature_mode=None, observed_mix=1., epochs=20, batch_size=8,
                      learning_rate=.02, identity_penalty=.001, gradient_clip=1., seed=2026090851,
                      cache_dir=None, row_batch_size=8, samples=None, bias_bound=4., normalization='none',
                      hidden_dim=0, scale_bound=2., beam_size=64, validation_every=None, validation_sources=None, on_epoch=None,
                      train_shapes=None, training_sources=None, train_per_shape=None,
                      early_stopping_patience=None, validation_shapes=None, training_device='cpu',
                      training_cache_bytes=1536 * 1024**2, device_cache_bytes=512 * 1024**2,
                      checkpoint_path=None, validation_device=None, tnorm='prod'):
    """Fit a frozen-backbone adapter; optionally select epochs on source validation."""
    sources = list(sources)
    if tnorm not in ('prod', 'min'):
        raise ValueError('Choose prod or min t-norm')
    device = torch.device(training_device)
    validation_device = torch.device(validation_device or device)
    cache_device = device if device.type == 'cuda' else validation_device
    if any(d.type not in ('cpu', 'cuda') for d in (device, validation_device)) or min(training_cache_bytes, device_cache_bytes) < 0:
        raise ValueError('Choose CPU/CUDA training and nonnegative cache budgets')
    if not sources or len({d.name for d in sources}) != len(sources):
        raise ValueError('Provide nonempty, uniquely named sources')
    if min(epochs, batch_size, row_batch_size) < 1 or not all(math.isfinite(v) for v in (learning_rate, identity_penalty, gradient_clip)):
        raise ValueError('Positive epoch/batch counts and finite optimizer settings required')
    if learning_rate <= 0 or identity_penalty < 0 or gradient_clip <= 0:
        raise ValueError('Invalid optimizer settings')
    if validation_every is not None and (type(validation_every) is not int or validation_every < 1):
        raise ValueError('validation_every must be a positive integer')
    if early_stopping_patience is not None:
        if type(early_stopping_patience) is not int or early_stopping_patience < 1:
            raise ValueError('early_stopping_patience must be a positive number of epochs')
        if validation_every is None:
            raise ValueError('Early stopping requires source validation checkpoint selection')
    train_shapes = tuple(train_shapes or TRAINING_SHAPES)
    if any(s not in TRAINING_SHAPES for s in train_shapes):
        raise ValueError('Unsupported training shape')
    validation_shapes = tuple(validation_shapes or TRAINING_SHAPES)
    if not validation_shapes or any(s not in TRAINING_SHAPES for s in validation_shapes):
        raise ValueError('Unsupported validation shape')
    if type(beam_size) is not int or beam_size < 1:
        raise ValueError('Positive beam_size required')
    training_names = set(training_sources) if training_sources is not None else {s.name for s in sources}
    if not training_names or training_names - {s.name for s in sources}:
        raise ValueError('Choose existing training sources')
    if train_per_shape is not None and (type(train_per_shape) is not int or train_per_shape < 1):
        raise ValueError('train_per_shape must be a positive integer')
    feature_mode = feature_mode or ('context_scores_v1' if getattr(model, 'name', '') == 'ULTRA' else 'context')
    with ExitStack() as resources:
        backbone = state_fingerprint(model)
        banks = []
        total_entities = sum(data.context.num_entities for data in sources)
        for data in sources:
            bank = _bank(model, data, cache_dir=cache_dir, row_batch_size=row_batch_size, seed=seed, samples=samples,
                         device=cache_device, cache_bytes=training_cache_bytes * data.context.num_entities // total_entities,
                         device_cache_bytes=device_cache_bytes * data.context.num_entities // total_entities)
            banks.append(bank)
            if 'provider' in bank:
                resources.callback(bank['provider'].close)
        for bank in banks:
            bank.update(beam_size=beam_size, tnorm=tnorm, validation_shapes=validation_shapes, device=device, validation_device=validation_device)
        selection_names = set(validation_sources) if validation_sources is not None else {s.name for s in sources}
        validation_indices = [i for i, s in enumerate(sources) if s.name in selection_names and s.validation]
        if validation_every is not None and (not validation_indices or selection_names - {s.name for s in sources}):
            raise ValueError('Checkpoint selection requires reserved source validation queries')
        settings = dict(objective='filtered_softmax_v1', epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                        identity_penalty=identity_penalty, gradient_clip=gradient_clip, seed=seed,
                        weighting='equal_source_shape', selection='source_validation_mrr' if validation_every else 'final_epoch',
                        shapes=sorted({q.shape for s in sources for q in s.train if q.shape in train_shapes}),
                        validation_sources=sorted(selection_names), validation_every=validation_every,
                        training_sources=sorted(training_names), train_per_shape=train_per_shape,
                        early_stopping_patience=early_stopping_patience, beam_size=beam_size, tnorm=tnorm, validation_shapes=list(validation_shapes),
                        device=str(device), validation_device=str(validation_device), cpu_threads=torch.get_num_threads(), training_cache_bytes=training_cache_bytes,
                        device_cache_bytes=device_cache_bytes)
        adapter = QueryScoreAdapter(feature_mode, observed_mix, metadata=dict(
            backbone_state_sha256=backbone, training=settings, score_banks=[bank['identity'] for bank in banks],
            sources={source.name: fingerprint(source.to_dict()) for source in sources}), bias_bound=bias_bound, scale_bound=scale_bound,
            normalization=normalization, hidden_dim=hidden_dim, seed=seed).to(device)
        prepared = [_prepared_bank(adapter, bank) for bank in banks]
        optimizer = torch.optim.Adam(adapter.parameters(), lr=learning_rate)
        examples = []
        selected_counts: dict[tuple[int, str], int] = {}
        for i, source in enumerate(sources):
            if source.name not in training_names:
                continue
            for query in source.train:
                cell = (i, query.shape)
                if query.shape in train_shapes and (train_per_shape is None or selected_counts.get(cell, 0) < train_per_shape):
                    examples.append((i, query))
                    selected_counts[cell] = selected_counts.get(cell, 0) + 1
        if not examples:
            raise ValueError('No queries for the selected training shapes')
        hard_answers = [q.answers - sources[i].context.answers(compile_query(q.query)) for i, q in examples]
        cells = {(i, q.shape) for i, q in examples}
        counts = {cell: sum((i, q.shape) == cell for i, q in examples) for cell in cells}
        rng, history, best = random.Random(seed), [], None
        stopped_early = False
        from .benchmark import _implementation_fingerprint
        resume_identity = fingerprint(dict(adapter=adapter.to_dict(), implementation=_implementation_fingerprint(),
                                           learning_rate=learning_rate, identity_penalty=identity_penalty,
                                           gradient_clip=gradient_clip, seed=seed))
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if saved['identity'] != resume_identity:
                raise ValueError('Adapter checkpoint inputs/settings changed')
            adapter.load_state_dict(saved['adapter'])
            optimizer.load_state_dict(saved['optimizer'])
            rng.setstate(saved['rng'])
            history, best = saved['history'], saved['best']
            stopped_early = history[-1].get('early_stopped', False) if history else False
        for epoch in range(epochs if stopped_early else len(history), epochs):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            order = list(range(len(examples)))
            rng.shuffle(order)
            epoch_loss = 0.
            for start in range(0, len(order), batch_size):
                optimizer.zero_grad()
                batch = order[start:start + batch_size]
                batch_terms = []
                for index in batch:
                    i, query = examples[index]
                    hard = hard_answers[index]
                    logs = _query_logs(adapter, prepared[i], query)
                    weight = len(examples) / (len(cells) * counts[i, query.shape])
                    term = weight * filtered_softmax_loss(logs, query.answers, hard) / len(batch)
                    term.backward()
                    batch_terms.append(term.detach())
                terms = torch.stack(batch_terms).tolist()
                if not all(math.isfinite(term) for term in terms):
                    raise FloatingPointError('Nonfinite adapter training objective')
                batch_loss = sum(terms)
                penalty = torch.cat([p.reshape(-1) for p in adapter.parameters()]).square().mean()
                loss = identity_penalty * penalty
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite adapter training objective')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                epoch_loss += batch_loss + float(loss.detach())
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            record = dict(epoch=epoch + 1, loss=epoch_loss / math.ceil(len(examples) / batch_size),
                          train_seconds=time.perf_counter() - started)
            if validation_every and ((epoch + 1) % validation_every == 0 or epoch + 1 == epochs):
                started = time.perf_counter()
                metrics = _validation(adapter, [sources[i] for i in validation_indices], [prepared[i] for i in validation_indices])
                record['validation_seconds'] = time.perf_counter() - started
                record['validation_mrr'] = sum(v['mrr'] for v in metrics.values()) / len(metrics)
                if not math.isfinite(record['validation_mrr']):
                    raise FloatingPointError('Nonfinite source-validation MRR')
                if best is None or record['validation_mrr'] > best[0]:
                    best = (record['validation_mrr'], epoch + 1,
                            {k: v.detach().clone() for k, v in adapter.state_dict().items()}, metrics)
                if early_stopping_patience is not None:
                    record.update(best_epoch=best[1], epochs_without_improvement=epoch + 1 - best[1])
                    stopped_early = epoch + 1 < epochs and epoch + 1 - best[1] >= early_stopping_patience
                    record['early_stopped'] = stopped_early
            history.append(record)
            if checkpoint_path is not None and ((epoch + 1) % (validation_every or 1) == 0 or epoch + 1 == epochs):
                _save_fit(checkpoint_path, dict(identity=resume_identity, adapter=adapter.state_dict(),
                          optimizer=optimizer.state_dict(), rng=rng.getstate(), history=history, best=best))
            if on_epoch is not None:
                on_epoch(record)
            if stopped_early:
                break
        if best is not None:
            adapter.load_state_dict(best[2])
            settings.update(selected_epoch=best[1], selected_validation_mrr=best[0])
        settings.update(epochs_completed=len(history), stopped_early=stopped_early,
                        stop_reason='validation_patience' if stopped_early else 'epoch_ceiling',
                        training_queries=len(examples), optimizer_steps=len(history) * math.ceil(len(examples) / batch_size))
        if state_fingerprint(model) != backbone:
            raise RuntimeError('Backbone changed during adapter fitting')
        settings['row_cache'] = {source.name: dict(bank['provider'].stats) for source, bank in zip(sources, banks) if 'provider' in bank}
        variants = {'sigmoid': QueryScoreAdapter('global').to(device), 'observed': QueryScoreAdapter('global', 1.).to(device)}
        validation = {name: _baseline_validation(value, sources, banks, cache_dir) for name, value in variants.items()}
        validation['fitted'] = best[3] if best is not None and len(validation_indices) == len(sources) else _validation(adapter, sources, banks)
        return AdapterFitResult(adapter, history, validation)
