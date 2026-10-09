"""Fit or build the adapters of the final studies from scratch.

The tracked successor of the scripts that made the shipped adapters
(Experiments/screens/refit.py, target_fit.py and the thresholds of brackets.py):
the same backbone settings, training data, seeds and fitting calls. Training
data is regenerated from the source and target graphs and checked against the
pinned fingerprints before the backbone is loaded. Refitting reproduces the
shipped weights up to floating-point differences of the recomputed backbone
scores (within 2e-6 for the reference adapters on H100), not bit for bit.
"""

import hashlib
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from dicee.models import KGICL, TRIX, ULTRA
from dicee.models._inference import float32_precision_backends
from dicee.query_answering import AdapterQuery, AdapterTrainingData, QueryContext, QueryScoreAdapter, fit_query_adapter, load_benchmark, prepare_adapter_data
from dicee.query_answering._query import compile_query
from dicee.query_answering.adapter_training import STANDARD_TRAINING_SHAPES, TRAINING_SHAPES
from dicee.query_answering.context import fingerprint, state_fingerprint

from ..manifests import REPO, read_manifest, suite_directory
from .registry import ADAPTER_ROOT, ALL_FITS, CHECKPOINTS, SEED, SOURCES, ULTRAQUERY_AS_ULTRA, ULTRAQUERY_RELEASE, sha256

VALIDATION_SHAPES = list(STANDARD_TRAINING_SHAPES)
# Source training data as the reference sweep prepared it (seed, mask and counts in each prepared file's metadata).
SOURCE_COUNTS = {'1p': 28, '2p': 28, '3p': 28, '2i': 140, '3i': 140, 'ip': 20, 'pi': 20, '2u': 20, 'up': 20, '2in': 70, '3in': 70,
                 'inp': 28, 'pin': 28, 'pni': 28}
TARGET_PER_SHAPE = 280  # One target graph supplies what the three sources supplied at 140 each.


def runtime() -> None:
    """Thread count, seed and float32 precision of every adapter fit."""
    torch.set_num_threads(4)
    torch.manual_seed(0)
    for backend in float32_precision_backends():
        backend.fp32_precision = 'ieee'


def load_backbone(backbone: str, input_root: Path, *, checkpoint: str | None = None, device: str = 'cuda',
                  relation_conditioning: str = 'direct'):
    """A frozen backbone as every fit and analysis loads it: query batches of 8; KG-ICL caches its relation tables."""
    extra = {'graph_projection_cache_mb': 512} if backbone == 'kgicl' else {}
    if relation_conditioning != 'direct':
        extra['ultra_relation_conditioning'] = relation_conditioning
    from dicee.models.flock import Flock
    model = {'ultra': ULTRA, 'trix': TRIX, 'kgicl': KGICL, 'flock': Flock}[backbone](dict(
        num_entities=1, num_relations=1, graph_inference_backend='auto', **{f'{backbone}_query_batch_size': 8}, **extra))
    if checkpoint == ULTRAQUERY_AS_ULTRA:
        ultraquery_as_ultra(input_root)
    path = Path(input_root) / (checkpoint or CHECKPOINTS[backbone])
    return model.load_pretrained(str(path)).to(device).eval().requires_grad_(False)


def ultraquery_as_ultra(input_root: Path) -> Path:
    """UltraQuery's released weights as a frozen ULTRA checkpoint: its 82 tensors without the ``model.model.`` prefix.

    The projection model of UltraQuery is ULTRA's architecture, so with ``relation_conditioning='query'`` the frozen
    ULTRA scores of these weights equal UltraQuery's single-source projection (checked in the tests).
    """
    target = Path(input_root) / ULTRAQUERY_AS_ULTRA
    if not target.is_file():
        source = Path(input_root) / ULTRAQUERY_RELEASE[0]
        if sha256(source) != ULTRAQUERY_RELEASE[1]:
            raise ValueError(f'{source} differs from the pinned UltraQuery release')
        state = torch.load(source, map_location='cpu', weights_only=True)['model']
        converted = {key.removeprefix('model.model.'): value for key, value in state.items()}
        if len(converted) != len(state) or not all(key.startswith(('relation_model.', 'entity_model.')) for key in converted):
            raise ValueError('Unexpected UltraQuery checkpoint layout')
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + '.partial')
        torch.save({'model': converted}, partial)
        os.replace(partial, target)
    return target


def source_data(name: str, input_root: Path, cache: Path) -> AdapterTrainingData:
    """The prepared training data of one adapter source graph, regenerated once and checked against its pin.

    Args:
        name: ``FB15k237``, ``WN18RR`` or ``CoDExMedium``.
        input_root: Root holding ``SOURCES[name]``'s training split.
        cache: Directory for the prepared JSON file.

    Raises:
        FileNotFoundError: If the training split is missing.
        ValueError: If the split or the regenerated data differs from the pinned bytes.
    """
    relative, digest, pinned = SOURCES[name]
    path = Path(cache) / f'{name}.json'
    if not path.is_file():
        train = Path(input_root) / relative
        if not train.is_file():
            raise FileNotFoundError(f'Missing adapter source {train} (SHA-256 {digest}); see python -m benchmarks.cqa fit --list')
        source_bytes = train.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != digest:
            raise ValueError(f'{train} differs from the pinned source graph')
        triples = [line.split() for line in source_bytes.decode().splitlines() if line.strip()]
        entities = {v: i for i, v in enumerate(sorted({v for h, _, t in triples for v in (h, t)}))}
        relations = {v: i for i, v in enumerate(sorted({r for _, r, _ in triples}))}
        count = len(relations)
        context = QueryContext(tuple((entities[h], relations[r], entities[t]) for h, r, t in triples),
                               len(entities), 2 * count, tuple((r, r + count) for r in range(count)))
        prepared = prepare_adapter_data(context, name=name, seed=2026090851, shapes=TRAINING_SHAPES, train_per_shape=140,
                                        validation_per_shape=16, train_counts=SOURCE_COUNTS, max_attempts=500_000)
        prepared.metadata.update(train_sha256=digest, source_path=relative)
        if fingerprint(prepared.to_dict()) != pinned:
            raise ValueError(f'Regenerated {name} training data differs from the pinned data')
        path.parent.mkdir(parents=True, exist_ok=True)
        prepared.save(path)
    # Fits read the saved file, as they read the reference sweep's prepared sources.
    data: AdapterTrainingData = AdapterTrainingData.load(path)
    if fingerprint(data.to_dict()) != pinned:
        raise ValueError(f'{path} differs from the pinned {name} training data')
    return data


def target_data(dataset: str, input_root: Path) -> AdapterTrainingData:
    """Training data of a target fit: queries sampled from the target's test inference graph with 30% of its pairs masked.

    +H targets use train+valid, UQ-23 targets their released test graph; no released target query or answer is read.
    """
    suite = 'plus_h' if dataset.endswith('+H') else 'ultraquery'
    data_root = Path(input_root) / read_manifest(suite_directory(suite) / 'kgfm_adapters.json')['data_root']
    context = load_benchmark(data_root, dataset, split='test').context
    # Preparation needs a training query of every prepared type; fitting uses only 2i/3i.
    counts = {shape: TARGET_PER_SHAPE if shape in ('2i', '3i') else 1 for shape in VALIDATION_SHAPES}
    data: AdapterTrainingData = prepare_adapter_data(context, name=dataset, seed=2026090851, shapes=VALIDATION_SHAPES, train_counts=counts,
                                                     validation_per_shape=16, max_attempts=500_000)
    return data


def target_validation_data(dataset: str, input_root: Path) -> AdapterTrainingData:
    """Training data of a target-validation fit: the target's released validation queries, never its test queries.

    At most 280 2i and 280 3i queries train; 16 queries of each of the ten validation types, disjoint from them,
    select the checkpoint. Queries are drawn with a fixed seed. Answers are the released easy and hard answers plus
    every proof in the validation inference graph; the hard answers that graph cannot prove are the positives.
    """
    suite = 'plus_h' if dataset.endswith('+H') else 'ultraquery'
    data_root = Path(input_root) / read_manifest(suite_directory(suite) / 'kgfm_adapters.json')['data_root']
    benchmark = load_benchmark(data_root, dataset, split='valid')
    context = benchmark.context
    by_shape = defaultdict(list)
    for item in sorted(benchmark.queries, key=lambda item: item.identity):
        by_shape[item.shape].append(item)
    rng = random.Random(SEED)
    train, validation, seen = [], [], set()
    for shape in VALIDATION_SHAPES:
        items = by_shape.get(shape, [])
        rng.shuffle(items)
        selected = []
        for item in items:
            if len(selected) == (16 + (TARGET_PER_SHAPE if shape in ('2i', '3i') else 0)):
                break
            proofs = context.answers(compile_query(item.query))
            answers = set(item.easy) | set(item.hard) | set(proofs)
            positives = set(item.hard) - set(proofs)
            if not positives or len(answers) >= context.num_entities:
                continue
            try:
                query = AdapterQuery(item.query, frozenset(answers), frozenset(positives))
            except ValueError:  # e.g. an intersection with repeated atoms, which training excludes
                continue
            if query.query in seen:
                continue
            seen.add(query.query)
            selected.append(query)
        validation.extend(selected[:16])
        train.extend(selected[16:])
    return AdapterTrainingData(dataset, context, tuple(train), tuple(validation),
                               metadata=dict(split='valid', seed=SEED, kind='target-valid'))


def fit(path: str, input_root: Path, output: Path, *, device: str = 'cuda') -> dict:
    """Fit or build the adapter ``ALL_FITS[path]`` into ``output/path`` and write its report next to it.

    Args:
        path: Adapter path relative to the adapter root, as in ``ALL_FITS``.
        input_root: Root with ``checkpoints/``, the source graphs (``KGs/``) and the suites' datasets.
        output: Adapter root of the fitted adapters; ``output/cache`` keeps training data, score banks and resumable state.
        device: Training device; source fits validate on the CPU, target fits on ``device``.

    Returns:
        The fit report.
    """
    spec, input_root, output = ALL_FITS[path], Path(input_root), Path(output)
    target, cache = output / path, output / 'cache'
    report_path = target.with_name(target.stem + '.report.json')
    if target.is_file():
        return json.loads(report_path.read_text()) if report_path.is_file() else {}
    runtime()
    start = time.monotonic()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')  # An adapter appears only once it is complete and checked.
    report: dict = dict(adapter=path, spec=asdict(spec))
    if spec.kind == 'threshold':
        # UltraQuery LP's fixed calibration, bound to the backbone weights; nothing is trained.
        model = load_backbone(spec.backbone, input_root, device='cpu')
        QueryScoreAdapter('global', 1., bias_bound=8., membership_threshold=spec.threshold, metadata=dict(
            backbone_state_sha256=state_fingerprint(model), calibration='ultraquery-lp-threshold',
            source='UltraQuery arXiv v2, Appendix B (benchmarks/ultraquery/comparisons.json)')).save(partial)
    elif spec.kind == 'calibration':
        # A training-free calibration of a published executor, bound to the backbone weights; nothing is trained.
        model = load_backbone(spec.backbone, input_root, checkpoint=spec.checkpoint, device='cpu',
                              relation_conditioning=spec.relation_conditioning)
        source = {'minmax': 'CQD-Hybrid (is-cqa-complex, max_norm 0.9)', 'softmax-degree': 'QTO (bys0318/QTO, calibrated rows)',
                  'softmax': 'QTO without the observed degree (review 2026-10-08)',
                  'softmax-degree-ties': 'QTO with ties at its cap ordered by the raw score (review 2026-10-08)'}
        QueryScoreAdapter('global', 1., bias_bound=8., fixed_calibration=spec.calibration, mask_known_logits=spec.mask_known,
                          metadata=dict(backbone_state_sha256=state_fingerprint(model), calibration=spec.calibration,
                                        source=source[spec.calibration] + ('; known-tail logits masked' if spec.mask_known else ''))
                          ).save(partial)
    else:
        if spec.kind == 'source':
            names = [name for name in SOURCES if name not in spec.without]
            if spec.sources:
                # Prepared source queries of another generator on the same masked graphs (e.g. hardness-balanced).
                data = [AdapterTrainingData.load(input_root / spec.sources / f'{name}.json') for name in names]
                report['sources_sha256'] = {name: sha256(input_root / spec.sources / f'{name}.json') for name in names}
            else:
                data = [source_data(name, input_root, cache / 'sources') for name in names]
            settings = dict(train_per_shape=140, validation_device='cpu', training_cache_bytes=6144 * 2**20)
        elif spec.kind == 'target-valid':
            names = [str(spec.target)]
            prepared = target_validation_data(names[0], input_root)
            provenance = cache / 'target-valid' / f'{names[0].replace(":", "-")}.json'
            provenance.parent.mkdir(parents=True, exist_ok=True)
            prepared.save(provenance)  # Provenance only.
            report.update(preparation_seconds=time.monotonic() - start, training_queries=len(prepared.train),
                          validation_queries=len(prepared.validation))
            data = [prepared]
            settings = dict(train_per_shape=TARGET_PER_SHAPE, validation_device=device, training_cache_bytes=2048 * 2**20)
        else:
            names = [str(spec.target)]
            prepared = target_data(names[0], input_root)
            # The shipped adapter records the fingerprint of the data it was fitted on.
            if fingerprint(prepared.to_dict()) != json.loads((REPO / ADAPTER_ROOT / path).read_text())['sources'][names[0]]:
                raise ValueError(f'Regenerated {names[0]} training data differs from the data of the shipped adapter')
            provenance = cache / 'targets' / f'{names[0].replace(":", "-")}.json'
            provenance.parent.mkdir(parents=True, exist_ok=True)
            prepared.save(provenance)  # Provenance only.
            report.update(preparation_seconds=time.monotonic() - start, masked_fact_pairs=prepared.metadata['masked_fact_pairs'],
                          training_queries=len(prepared.train))
            data = [prepared]
            settings = dict(train_per_shape=TARGET_PER_SHAPE, validation_device=device, training_cache_bytes=2048 * 2**20)
        model = load_backbone(spec.backbone, input_root, checkpoint=spec.checkpoint, device=device,
                              relation_conditioning=spec.relation_conditioning)
        result = fit_query_adapter(
            model, data, feature_mode=spec.feature_mode, bias_bound=spec.bias_bound, scale_bound=spec.scale_bound,
            observed_mix=spec.observed_mix, epochs=500, train_shapes=['2i', '3i'],
            training_sources=names, validation_sources=names, validation_every=5, validation_shapes=VALIDATION_SHAPES,
            early_stopping_patience=100, seed=spec.seed, row_batch_size=8, cache_dir=cache / 'score-banks' / spec.backbone,
            training_device=device, device_cache_bytes=512 * 2**20, checkpoint_path=cache / 'state' / f'{path.removesuffix(".json")}.pt',
            **settings)
        result.adapter.save(partial)
        training = result.adapter.metadata['training']
        report.update(selected_epoch=training['selected_epoch'], selected_validation_mrr=training['selected_validation_mrr'],
                      history=result.history)
    QueryScoreAdapter.load(partial, model=model)
    report.update(seconds=time.monotonic() - start, adapter_sha256=sha256(partial))
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    os.replace(partial, target)
    print(json.dumps({key: report[key] for key in ('adapter', 'seconds', 'adapter_sha256')}), flush=True)
    return report


def fit_all(paths: list[str], input_root: Path, output: Path, *, device: str = 'cuda') -> None:
    """Fit ``paths`` in order in this process; finished adapters are kept, interrupted fits resume."""
    for path in paths:
        if path not in ALL_FITS:
            raise ValueError(f'No fit recipe for {path!r}; see python -m benchmarks.cqa fit --list')
    for path in paths:
        fit(path, input_root, output, device=device)
