"""Analyses the thesis cites besides the benchmark tables; each returns a JSON-ready report.

    python -m benchmarks.cqa analysis NAME [options] --output FILE.json

- ``calibration-profile``: raw and adapted memberships of frozen backbone atoms on 300 validation ``1p`` queries
  per graph (scale and shift, non-answer mass, expected calibration error).
- ``observed-links``: how many true answers of intersection queries have an observed link in the test graph.
- ``adapter-weights``: the adapter weights of the five training seeds per backbone.
- ``negation-probe``: why uncalibrated negation collapses: raw rows and ``2in`` queries without beam pruning.
- ``pretraining-overlap``: test and graph triples of every target that occur in a backbone's pretraining graphs.
- ``answer-classes``: FB15k and FB15k237 test answers of 1p/2p queries by their cheapest grounding (missing links,
  which hop, and whether the missing pair is connected in the training graph), with each method's answer-level MRR
  from the frozen rank traces.

Only validation queries, test graphs, test answer sets and completed rank traces are read; no setting is chosen on
test data.
"""

import hashlib
import json
import math
import pickle
import random
import sqlite3
import time
import zipfile
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

from ..manifests import REPO, catalog, read_manifest, suite_directory
from .registry import ADAPTER_ROOT, BACKBONES, SOURCES

FEATURES = ('constant', 'observed_tails', 'head_degree', 'relation_frequency', 'mean_score', 'score_entropy', 'top_two_gap', 'observed_contrast')
INTERSECTIONS = ('2i', '3i', '2in', '3in')
# KG-ICL's pretraining graphs (GraIL fb237_v1 and nell_v1, CoDEx-S) in datasets.zip of the pinned KG-ICL checkout.
KGICL_PRETRAINING = {
    'fb237_v1': {'train': ('inductive/fb237_v1/train.txt', '74ef3120fe38a22e5131064d23108b88a8d339496057fa8467d80dcda8f1cec1'),
                 'valid': ('inductive/fb237_v1/valid.txt', 'd4c2698d7dc36c0dc39ffb3d810caeafe530d779bf9f6451fac2027f41c2ff6d')},
    'nell_v1': {'train': ('inductive/nell_v1/train.txt', 'd2b5d535ace4d1e3a18e69176e84a3ada1ca62fb4a647a12c4222781b91615ba'),
                'valid': ('inductive/nell_v1/valid.txt', 'd11d6de249ff1185e701536a7d63f0c997cf7cce2646f3d17c06ecc885835726')},
    'codex-s': {'train': ('transductive/codex-s/train.txt', '64f93b7f314f3936a6f65739721429db3f6a7c8f5a1e1104ec3bb544f7434f59'),
                'valid': ('transductive/codex-s/valid.txt', '3831c0e57daef03c3a18cdd1a72e370b496f696c5218883d35c7d2ab8a6a772c')},
}
# Rank traces of the answer-class analysis: method -> (study ID, entry directory below STUDY/test).
ANSWER_CLASS_TRACES = {
    'UltraQuery': ('ultraquery-baselines', 'ultraquery-{dataset}'),
    'UltraQuery LP': ('ultraquery-pergraph', 'ultraquery-lp-{dataset}'),
    'QTO': ('ultraquery-qto', 'qto-{dataset}'),
    'ULTRA raw': ('ultraquery-kgfm-b64', 'ultra-product-intersections-{dataset}/without-adapter'),
    'ULTRA+ad': ('ultraquery-kgfm-b64', 'ultra-product-intersections-{dataset}'),
    'TRIX+ad': ('ultraquery-kgfm-b64', 'trix-product-intersections-{dataset}'),
    'KG-ICL+ad': ('ultraquery-kgfm-b64-kgicl', 'kgicl-product-intersections-{dataset}'),
}


def datasets(names: list[str] | None = None) -> list[tuple[str, str, Path]]:
    """(dataset, suite, data root below the input root) of the 23 UQ-23 and three +H datasets, or of ``names``."""
    roots = {suite: Path(read_manifest(suite_directory(suite) / 'kgfm_adapters.json')['data_root']) for suite in ('ultraquery', 'plus_h')}
    every = [(name, 'ultraquery', roots['ultraquery']) for name in catalog().BENCHMARK_DATASETS]
    every += [(name, 'plus_h', roots['plus_h']) for name in catalog().PLUS_H_DATASETS]
    if names is not None and set(names) - {name for name, _, _ in every}:
        raise ValueError(f'Unknown datasets: {sorted(set(names) - {name for name, _, _ in every})}')
    return [item for item in every if names is None or item[0] in names]


def _quantiles(values) -> dict:
    """10th, 50th and 90th percentile; None for no values."""
    import torch
    if not values.numel():
        return dict(p10=None, median=None, p90=None)
    q = torch.quantile(values, torch.tensor([.1, .5, .9], dtype=values.dtype))
    return dict(p10=q[0].item(), median=q[1].item(), p90=q[2].item())


def _statistic(values, name: str) -> float | None:
    """``values.mean()`` or ``values.median()``; None for no values."""
    return getattr(values, name)().item() if values.numel() else None


def _scale_and_shift(adapter, features):
    """The adapter's per-row scale and shift, exactly as QueryScoreAdapter.transform computes them."""
    u, v = (features @ adapter.weights.detach().T).unbind(1)
    log_scale = math.log(2) * u
    if adapter.scale_bound is not None:
        limit = math.log(adapter.scale_bound)
        log_scale = limit * (u * (math.log(2) / limit)).tanh()
    return log_scale.exp(), adapter.bias_bound * v.tanh()


def _calibration(memberships, hard, keep, bins=10) -> dict:
    """Mean membership of hard answers and of others, expected non-answers per row, and ECE over equal-width bins."""
    values, labels = memberships[keep], hard[keep]
    index = (values * bins).long().clamp(max=bins - 1)
    ece = 0.
    for b in range(bins):
        mask = index == b
        if mask.any():
            ece += mask.sum().item() / len(values) * abs(values[mask].mean().item() - labels[mask].double().mean().item())
    others = keep & ~hard
    return dict(hard_answer_membership=memberships[keep & hard].mean().item(), other_membership=memberships[others].mean().item(),
                expected_non_answers_per_atom=(memberships * others).sum(1).mean().item(), ece=ece)


def calibration_profile(backbone: str, input_root: Path, *, names: list[str] | None = None, atoms: int = 300, device: str = 'cuda',
                        adapter_root: Path = REPO / ADAPTER_ROOT) -> dict:
    """Per-graph calibration of frozen atom scores, raw (identity control) and with the source-only adapter.

    For every dataset, sample validation ``1p`` queries (anchor h, relation r), score every entity on the validation
    inference graph and compare memberships without the adapter (sigmoid of the raw score, observed facts set to one)
    and with it: the adapter's per-atom scale and shift, the mean membership of hard answers and of other entities, the
    expected number of non-answers per atom, and the expected calibration error over ten membership bins. Easy answers
    (observed facts) are left out of every statistic. The adapter is monotone per row, so 1p rankings do not change.

    Args:
        backbone: ``ultra``, ``trix`` or ``kgicl``.
        input_root: Root with ``checkpoints/`` and the suites' datasets.
        names: Datasets (default: all 26).
        atoms: Validation 1p queries sampled per dataset (seed 0).
        device: Backbone device.
        adapter_root: Directory of ``BACKBONE_product_intersections.json``.
    """
    import torch

    from dicee.query_answering import QueryScoreAdapter, load_benchmark
    from dicee.query_answering.context import attached_context

    from .fit import load_backbone, runtime
    runtime()
    model = load_backbone(backbone, input_root, device=device)
    adapter_path = Path(adapter_root) / f'{backbone}_product_intersections.json'
    adapter = QueryScoreAdapter.load(adapter_path, model=model)
    identity = QueryScoreAdapter('global', adapter.observed_mix)
    shown = adapter_path.relative_to(REPO).as_posix() if adapter_path.is_relative_to(REPO) else str(adapter_path)
    report: dict = dict(backbone=backbone, adapter=shown, atoms=atoms, split='valid', datasets={})
    for dataset, suite, root in datasets(names):
        start = time.monotonic()
        data = load_benchmark(Path(input_root) / root, dataset, split='valid', query_types=['1p'])
        queries = [q for q in data.queries if q.shape == '1p']
        sample = random.Random(0).sample(queries, min(atoms, len(queries)))
        conditions = [(q.query[0], q.query[1][0]) for q in sample]
        with attached_context(model, data.context), torch.no_grad():
            raw = torch.cat([model.forward_k_vs_all(torch.tensor(conditions[i:i + 8]))
                             for i in range(0, len(conditions), 8)]).double().cpu()
        observed, base = data.context.features(conditions, device='cpu')
        hard = torch.zeros((len(sample), data.context.num_entities), dtype=torch.bool)
        easy = torch.zeros_like(hard)
        for i, q in enumerate(sample):
            hard[i, sorted(q.hard)] = True
            easy[i, sorted(q.easy)] = True
        keep = ~(easy | observed)
        _, _, features = adapter.prepare(raw, observed, base)
        scale, shift = _scale_and_shift(adapter, features)
        with torch.no_grad():
            learned = adapter.transform(raw, observed, features).exp()
            control = identity.transform(*identity.prepare(raw, observed, base)).exp()
        row: dict = dict(
            suite=suite, entities=data.context.num_entities, atoms=len(sample), scale=_quantiles(scale), shift=_quantiles(shift),
            without_adapter=_calibration(control, hard, keep), with_adapter=_calibration(learned, hard, keep),
            seconds=round(time.monotonic() - start, 1))
        report['datasets'][dataset] = row
        print(f'{dataset}: scale {row["scale"]["median"]:.2f}, shift {row["shift"]["median"]:+.2f}; '
              f'ECE {row["without_adapter"]["ece"]:.4f} -> {row["with_adapter"]["ece"]:.4f}; expected non-answers/atom '
              f'{row["without_adapter"]["expected_non_answers_per_atom"]:.1f} -> {row["with_adapter"]["expected_non_answers_per_atom"]:.1f}',
              flush=True)
    return report


def observed_links(input_root: Path, *, names: list[str] | None = None) -> dict:
    """How often true answers and wrong candidates of intersection queries have an observed link, per test graph.

    For anchored intersections (2i, 3i and the positive atoms of 2in, 3in), count the positive atoms whose link to a
    candidate is an observed fact of the test-time graph (train+valid for +H). With observed facts on, such an atom
    gives membership 1.
    """
    from dicee.query_answering import load_benchmark
    report: dict = {}
    for dataset, suite, root in datasets(names):
        data = load_benchmark(Path(input_root) / root, dataset, split='test', inference_graph='train+valid' if suite == 'plus_h' else None,
                              query_types=list(INTERSECTIONS))
        observed: dict = defaultdict(set)
        for h, r, t in data.context.triples:
            observed[h, r].add(t)
        stats: dict = defaultdict(Counter)
        for query in data.queries:
            hits: Counter = Counter()
            for anchor, relations in query.query:
                if relations[-1] != -2:
                    for entity in observed[anchor, relations[0]]:
                        hits[entity] += 1
            counts = stats[query.shape]
            counts['queries'] += 1
            counts['answers'] += len(query.hard)
            counts['answers_with_observed_link'] += sum(hits[answer] > 0 for answer in query.hard)
            counts['wrong_candidates_with_observed_link'] += sum(e not in query.hard and e not in query.easy for e in hits)
        report[dataset] = {shape: dict(stats[shape], share=stats[shape]['answers_with_observed_link'] / stats[shape]['answers'])
                           for shape in INTERSECTIONS if stats[shape]['queries']}
        print(f'{dataset}: ' + '; '.join(f'{shape} {values["share"]:.0%}' for shape, values in report[dataset].items()), flush=True)
        del data
    return report


def adapter_weights(adapter_root: Path = REPO / ADAPTER_ROOT) -> dict:
    """Mean, range and sample s.d. of each adapter weight over the five training seeds (reference and seeds 1-4) per backbone.

    Row 0 of the weights sets the log-scale, row 1 the additive shift; features are those of ``context_scores``.
    """
    report: dict = {}
    for backbone in BACKBONES:
        paths = [f'{backbone}_product_intersections.json', *(f'seeds/{backbone}_product_intersections_seed{seed}.json' for seed in (1, 2, 3, 4))]
        weights = [json.loads((Path(adapter_root) / path).read_text())['weights'] for path in paths]
        report[backbone] = {'adapters': paths}
        for row, label in ((1, 'shift'), (0, 'scale')):
            values = {name: [w[row][i] for w in weights] for i, name in enumerate(FEATURES)}
            report[backbone][label] = {name: dict(mean=sum(v) / len(v), min=min(v), max=max(v),
                                                  sd=math.sqrt(sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)))
                                       for name, v in values.items()}
        shift = report[backbone]['shift']
        print(f'{backbone}: shift observed tails {shift["observed_tails"]["mean"]:+.2f}, entropy {shift["score_entropy"]["mean"]:+.2f}, '
              f'top-two gap {shift["top_two_gap"]["mean"]:+.2f}', flush=True)
    return report


def negation_probe(input_root: Path, *, backbone: str = 'kgicl', names: list[str] | None = None, atoms: int = 160, queries: int = 80,
                   device: str = 'cpu', adapter_root: Path = REPO / ADAPTER_ROOT) -> dict:
    """Why uncalibrated negation collapses: raw score rows of validation 1p atoms and 2in queries, with and without the adapter.

    A 2in query (a, r1) and not (b, r2) is scored m1 * (1 - m2), the product t-norm without beam pruning. Entities a
    backbone never reaches get raw logit exactly 0, so membership 0.5 without the adapter.
    """
    import torch

    from dicee.query_answering import QueryScoreAdapter, load_benchmark
    from dicee.query_answering.context import attached_context

    from .fit import load_backbone, runtime
    runtime()
    model = load_backbone(backbone, input_root, device=device)
    adapter = QueryScoreAdapter.load(Path(adapter_root) / f'{backbone}_product_intersections.json', model=model)
    identity = QueryScoreAdapter('global', 1.)

    def rows(data, conditions):
        with attached_context(model, data.context), torch.no_grad():
            raw = torch.cat([model.forward_k_vs_all(torch.tensor(conditions[i:i + 8]).to(device)).cpu()
                             for i in range(0, len(conditions), 8)]).double()
        observed, base = data.context.features(conditions, device='cpu')
        with torch.no_grad():
            plain = identity.transform(*identity.prepare(raw, observed, base)).exp()
            learned = adapter.transform(*adapter.prepare(raw, observed, base)).exp()
        return raw, observed, plain, learned

    report: dict = dict(backbone=backbone, atoms=atoms, queries=queries, split='valid', datasets={})
    for dataset, _, root in datasets(names or ['FB15k237LogicalQuery', 'NELL995LogicalQuery', 'WikiTopicsQuery:art']):
        data = load_benchmark(Path(input_root) / root, dataset, split='valid', query_types=['1p', '2in'])
        n, rng = data.context.num_entities, random.Random(0)
        sample = rng.sample([x for x in data.queries if x.shape == '1p'], min(atoms, sum(x.shape == '1p' for x in data.queries)))
        raw, observed, plain, learned = rows(data, [(x.query[0], x.query[1][0]) for x in sample])
        hard = torch.zeros_like(observed)
        for i, x in enumerate(sample):
            hard[i, sorted(x.hard)] = True
        others, zero = ~hard & ~observed, raw == 0
        row: dict = dict(entities=n, atoms=len(sample), never_reached=zero.double().mean().item(), never_reached_hard_answers=_statistic(zero[hard].double(), 'mean'),
                   raw_logits_reached_non_answers=_quantiles(raw[others & ~zero]), raw_logits_reached_hard_answers=_quantiles(raw[hard & ~zero]),
                   never_reached_membership=dict(raw=_statistic(plain[zero], 'mean'), adapted_median=_statistic(learned[zero], 'median')),
                   reached_non_answer_membership_median=dict(raw=_statistic(plain[others & ~zero], 'median'),
                                                             adapted=_statistic(learned[others & ~zero], 'median')),
                   hard_answers_below_half_raw=_statistic((plain[hard] < .5).double(), 'mean'))
        twoin = rng.sample([x for x in data.queries if x.shape == '2in'], min(queries, sum(x.shape == '2in' for x in data.queries)))
        pairs = []
        for x in twoin:
            (a, (r1,)), (b, (r2, _)) = x.query
            pairs += [(a, r1), (b, r2)]
        raw2, _, plain2, learned2 = rows(data, pairs)
        for label, member in (('raw', plain2), ('adapted', learned2)):
            reciprocal, top_unreached, answer_keep, top_keep = [], [], [], []
            for i, x in enumerate(twoin):
                m1, m2 = member[2 * i], member[2 * i + 1]
                score = m1 * (1 - m2)
                excluded = torch.zeros(n, dtype=torch.bool)
                excluded[sorted(x.easy | x.hard)] = True
                for answer in x.hard:
                    reciprocal.append(1 / (1 + ((score > score[answer]) & ~excluded).sum().item()))
                top = score.masked_fill(excluded, -1).topk(min(10, n)).indices
                top_unreached.append((raw2[2 * i + 1][top] == 0).double().mean().item())
                answer_keep.append((1 - m2[sorted(x.hard)]).mean().item())
                top_keep.append((1 - m2[top]).mean().item())
            row[f'2in_{label}'] = dict(mrr=sum(reciprocal) / len(reciprocal), hard_answer_keep=sum(answer_keep) / len(answer_keep),
                                       top10_keep=sum(top_keep) / len(top_keep), top10_never_reached=sum(top_unreached) / len(top_unreached))
        report['datasets'][dataset] = row
        print(f'{dataset}: 2in MRR raw {row["2in_raw"]["mrr"]:.3f}, adapted {row["2in_adapted"]["mrr"]:.3f}; 1 - m(negated) for hard answers '
              f'{row["2in_raw"]["hard_answer_keep"]:.2f} raw, {row["2in_adapted"]["hard_answer_keep"]:.2f} adapted', flush=True)
    return report


def _canonical(h, r, t):
    """One direction per fact: reciprocal relations (-r, _reverse, _inverse, _inv) are flipped, +r loses its sign."""
    if r.startswith('-'):
        return t, r[1:], h
    if r.startswith('+'):
        return h, r[1:], t
    for suffix in ('_reverse', '_inverse', '_inv'):
        if r.endswith(suffix):
            return t, r[:-len(suffix)], h
    return h, r, t


def _named(path: Path, entities: dict, relations: dict) -> set | None:
    """Canonical named triples of an integer triple file, or None if the file is missing."""
    if not path.is_file():
        return None
    triples = set()
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3:
            h, r, t = map(int, parts)
            triples.add(_canonical(entities[h], relations[r], entities[t]))
    return triples


def _mapping(folder: Path, entity_key: str) -> tuple[dict, dict]:
    """id -> name of entities and relations from a dataset's mapping pickles."""
    if (folder / 'og_mappings.pkl').is_file():
        mappings = pickle.loads((folder / 'og_mappings.pkl').read_bytes())
        entities, relations = mappings[entity_key], mappings['r2id']
        return ({v: k for k, v in entities.items()} if not isinstance(next(iter(entities)), int) else entities,
                {v: k for k, v in relations.items()} if not isinstance(next(iter(relations)), int) else relations)
    return pickle.loads((folder / 'id2ent.pkl').read_bytes()), pickle.loads((folder / 'id2rel.pkl').read_bytes())


def target_triples(dataset: str, input_root: Path) -> dict:
    """Named splits of a target: train/valid/test of the transductive and +H graphs; the test inference graph
    (``graph``) and the test links (``test``) of the inductive ones. A missing file maps to None."""
    from dicee.query_answering.datasets import dataset_spec
    (_, _, data_root), = datasets([dataset])
    setting, name, _ = dataset_spec(dataset)
    folder = Path(input_root) / data_root / name
    if setting == 'transductive':
        entities, relations = _mapping(folder, 'e2id')
        graph = folder / 'KG_splits' if dataset == 'ICEWS18+H' else folder
        return {split: _named(graph / f'{split}.txt', entities, relations) for split in ('train', 'valid', 'test')}
    entities, relations = _mapping(folder, 'e2id' if setting == 'inductive-e' else 'e2id_test')
    test = 'test_predict.txt' if setting == 'inductive-e' else 'test_prediction.txt'
    return dict(graph=_named(folder / 'test_inference.txt', entities, relations), test=_named(folder / test, entities, relations))


def _pretraining(pretraining: str, input_root: Path, kgicl_datasets: Path | None) -> dict:
    """{graph: {split: set of raw triples}} of a backbone's pretraining graphs, checked against their pinned bytes."""
    def triples(raw: bytes, nell: bool = False) -> set:
        result = set()
        for line in raw.decode().splitlines():
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) == 3:
                h, r, t = parts
                if nell:  # GraIL names NELL entities concept:type:name, BetaE concept_type_name.
                    h, t = h.replace(':', '_'), t.replace(':', '_')
                result.add((h, r, t))
        return result

    def checked(raw: bytes, name: str, digest: str) -> bytes:
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f'{name} differs from the pinned pretraining graph')
        return raw

    if pretraining == 'ultra':  # ULTRA 3g and TRIX: the training splits of FB15k237, WN18RR and CoDEx-Medium.
        return {name: {'train': triples(checked((Path(input_root) / path).read_bytes(), path, digest))}
                for name, (path, digest, _) in SOURCES.items()}
    if kgicl_datasets is None:
        raise ValueError("KG-ICL's pretraining graphs need --kgicl-datasets, datasets.zip of the pinned KG-ICL checkout")
    with zipfile.ZipFile(kgicl_datasets) as archive:
        return {name: {split: triples(checked(archive.read(member), member, digest), nell=name == 'nell_v1')
                       for split, (member, digest) in splits.items()} for name, splits in KGICL_PRETRAINING.items()}


def pretraining_overlap(input_root: Path, *, pretraining: str = 'ultra', kgicl_datasets: Path | None = None,
                        names: list[str] | None = None) -> dict:
    """Exact overlap of every target's splits with a backbone's pretraining graphs, by original identifiers.

    Args:
        input_root: Root with the suites' datasets (and, for ``ultra``, the adapter source graphs below ``KGs/``).
        pretraining: ``ultra`` (ULTRA 3g and TRIX) or ``kgicl``.
        kgicl_datasets: ``datasets.zip`` of the pinned KG-ICL checkout (``kgicl`` only).
        names: Target datasets (default: all 26).

    Returns:
        Per target: the size of each split and, per pretraining graph and split (train, and train+valid where the
        graph has one), the triples each target split shares with it, the shared entities, and how many of the
        pretraining triples occur anywhere in the target.
    """
    graphs = _pretraining(pretraining, input_root, kgicl_datasets)
    report: dict = dict(pretraining=pretraining, graphs={name: {split: len(t) for split, t in splits.items()} for name, splits in graphs.items()},
                  datasets={})
    for dataset, _, _ in datasets(names):
        splits = target_triples(dataset, input_root)
        present = {split: values for split, values in splits.items() if values is not None}
        target = set().union(*present.values()) if present else set()
        entities = {e for h, _, t in target for e in (h, t)}
        row: dict = dict(sizes={split: len(values) if values is not None else None for split, values in splits.items()}, overlap={})
        for name, parts in graphs.items():
            variants = {'train': parts['train'], **({'train+valid': parts['train'] | parts['valid']} if 'valid' in parts else {})}
            row['overlap'][name] = {variant: dict({split: len(values & graph) for split, values in present.items()},
                                                  pretraining_in_target=len(graph & target),
                                                  shared_entities=len(entities & {e for h, _, t in graph for e in (h, t)}))
                                    for variant, graph in variants.items()}
        report['datasets'][dataset] = row
        test = present.get('test')
        cells = [f'{name} {row["overlap"][name]["train"]["test"] / len(test):.2%}' for name in graphs if test]
        print(f'{dataset}: test links in pretraining train: ' + (', '.join(cells) if test else 'test links missing'), flush=True)
    return report


def answer_class(shape: str, query: tuple, hard: frozenset, full: dict, train: dict, pairs: set) -> dict:
    """Class of every hard answer of a 1p or 2p query by its cheapest grounding in the complete graph.

    A class is (missing links, ``hop 1``/``hop 2``/``both`` for where they are, whether every missing link's entity
    pair is connected in the training graph by another relation or direction). Among equally cheap groundings, one
    whose missing pairs are connected is preferred, as such a link is the easiest to recover.
    """
    if shape == '1p':
        anchor, _ = query
        return {t: ('1 missing', 'hop 1', (anchor, t) in pairs) for t in hard}
    anchor, (r1, r2) = query
    best: dict = {}
    for y in full[anchor].get(r1, ()):
        first = y not in train[anchor].get(r1, ())
        for t in hard.intersection(full[y].get(r2, ())):
            second = t not in train[y].get(r2, ())
            hop = 'hop 1' if first and not second else 'hop 2' if second and not first else 'both'
            connected = all(pair in pairs for missing, pair in ((first, (anchor, y)), (second, (y, t))) if missing)
            key = (first + second, -connected)
            if t not in best or key < best[t][0]:
                best[t] = (key, (f'{first + second} missing', hop, connected))
    return {t: value for t, (_, value) in best.items()}


def answer_classes(input_root: Path, results: Path, *, names: list[str] | None = None) -> dict:
    """Answer-level MRR (sort ties) of every method per answer class of the 1p/2p test queries of transductive UQ-23 targets.

    FB15k keeps reverse and duplicate relations of its test links in the training graph, which FB15k237 removed; the
    classes separate answers whose missing links are such recoverable pairs (reverse edge observed) from the others.

    Args:
        input_root: Root of the UQ-23 datasets.
        results: Root of the study directories (``STUDY/test/ENTRY/ranks.sqlite3``), as ``reproduce`` writes them.
        names: Transductive UQ-23 datasets (default: FB15kLogicalQuery and FB15k237LogicalQuery).

    Returns:
        Per dataset: the rank trace of each method (None if missing) and, per shape and class, the number of hard
        answers and each method's answer-level MRR.
    """
    from dicee.query_answering.datasets import dataset_spec

    from ..difficulty import load_graphs, root_queries
    report: dict = dict(results=str(results), methods=list(ANSWER_CLASS_TRACES), datasets={})
    for dataset, _, root in datasets(names or ['FB15kLogicalQuery', 'FB15k237LogicalQuery']):
        if dataset not in catalog().TRANSDUCTIVE:
            raise ValueError(f'Answer classes need a transductive UQ-23 dataset, whose test graph is its training graph: {dataset}')
        folder = Path(input_root) / root / dataset_spec(dataset)[1]
        full, graphs, _, _ = load_graphs(folder)
        train = graphs['train']
        pairs = {(h, t) for h in train for r in train[h] for t in train[h][r]}
        records = root_queries(folder, ['1p', '2p'])
        classes = {key: answer_class(r['shape'], r['query'], r['hard'], full, train, pairs) for key, r in records.items()}
        answers = Counter((records[key]['shape'], label) for key, labels in classes.items() for label in labels.values())
        reciprocal: dict = defaultdict(lambda: defaultdict(list))
        traces = {}
        for method, (study, entry) in ANSWER_CLASS_TRACES.items():
            trace = Path(results) / study / 'test' / entry.format(dataset=dataset) / 'ranks.sqlite3'
            traces[method] = str(trace) if trace.is_file() else None
            if trace.is_file():
                with closing(sqlite3.connect(f'{trace.resolve().as_uri()}?mode=ro', uri=True)) as db:
                    for key, shape, payload in db.execute('SELECT query_id, shape, answers FROM queries'):
                        if shape in ('1p', '2p') and key in classes:
                            for answer, rank, *_ in json.loads(payload):
                                reciprocal[shape, classes[key].get(answer, ('unclassified', '', False))][method].append(1 / rank)
        rows = []
        for shape, label in sorted(answers.keys() | reciprocal.keys(), key=lambda item: (item[0], str(item[1]))):
            missing, hop, connected = label
            rows.append(dict(shape=shape, missing=missing, hop=hop, reverse_edge_observed=connected, answers=answers.get((shape, label), 0),
                             mrr={method: sum(values) / len(values) for method, values in reciprocal[shape, label].items()}))
        report['datasets'][dataset] = dict(traces=traces, classes=rows)
        print(f'== {dataset}' + ''.join(f'; {method}: trace missing' for method, trace in traces.items() if trace is None))
        print('class'.ljust(46) + 'answers'.rjust(8) + ''.join(method[:11].rjust(12) for method in ANSWER_CLASS_TRACES))
        for row in rows:
            label = f'{row["shape"]} {row["missing"]}, {row["hop"]}, reverse edge {"observed" if row["reverse_edge_observed"] else "absent"}'
            print(label.ljust(46) + str(row['answers']).rjust(8) + ''.join(
                f'{100 * row["mrr"][method]:12.1f}' if method in row['mrr'] else '           -' for method in ANSWER_CLASS_TRACES), flush=True)
    return report
