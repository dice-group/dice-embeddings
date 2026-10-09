"""The final CQA studies of the paper and thesis, and how every adapter they use was made.

A study is one frozen ``prepare`` of the 2026-10 final runs: recipe files of
its suite, an entry selection, an optional calibration bracket and the
``prepare`` flags. ``recipe_manifest`` rebuilds the manifest that ``prepare``
read, and ``resolved_manifest`` the manifest it froze. ``FITS`` records how
each shipped adapter below ``benchmarks/adapters`` was fitted or built, so the
``fit`` stage can regenerate it. Everything here reads tracked JSON only, so
the launcher plans and prints a reproduction without PyTorch.
"""

import copy
import fnmatch
import hashlib
import json
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from ..manifests import REPO, SUITES, catalog, prepare_manifest, read_manifest, suite_directory

ADAPTER_ROOT = 'benchmarks/adapters'
BACKBONES = ('ultra', 'trix', 'kgicl')
CHECKPOINTS = {'ultra': 'checkpoints/ultra_3g.pth', 'trix': 'checkpoints/trix/entity_prediction.pth',
               'kgicl': 'checkpoints/kgicl/KG-ICL-6L/model_best.tar', 'flock': 'checkpoints/flock/flock_entity.pth'}
SEED = 2026090851
# The reference KG-ICL recipe and its training-seed replicates in kgfm_kgicl.json.
KGICL, KGICL_SEEDS = ('kgicl-product-intersections-',), ('kgicl-product-intersections-seed',)
# Reference points registered on 2026-10-05, before their test runs. The notes are those of the frozen manifests;
# refit.py and target_fit.py, the scripts they name, became ``python -m benchmarks.cqa fit``.
BRACKET_STATUS = 'registered reference point (2026-10-05); evaluated once on test; cannot change the recipe'
BRACKETS = {
    'global': 'Global 2-parameter adapter (one scale and one shift for every atom), fitted like the shipped recipe '
              '(refit.py --feature-mode global, seed 2026090851). Lower reference point.',
    'uqlp': "UltraQuery LP's per-dataset membership thresholds (arXiv v2, Appendix B) as a fixed calibration: identity "
            'adapter with membership_threshold, observed facts on. Lower reference point.',
    'target-fit': "One adapter per target, fitted on queries sampled from the target's test inference graph with 30% of "
                  'its fact pairs masked (target_fit.py --split test); no released target query or answer is read. '
                  'Upper reference point.',
}
# Review experiments, registered on 2026-10-08 before their test runs (Experiments/screens/PROTOCOL.md).
REVIEW_STATUS = 'review experiment registered 2026-10-08 before its test run; evaluated once on test; cannot change the recipe'
REVIEW = {
    'minmax': "CQD-Hybrid's training-free calibration: 0.9 times the min-max normalized row, observed facts at one. "
              'No fitted parameter. Training-free reference point.',
    'softmax-degree': "QTO's training-free calibration: softmax over tails times the observed tail count (at least one), "
                      'capped at 0.9999, observed facts at one. No fitted parameter. Training-free reference point.',
    'gamma0': 'Adapter refitted with observed facts off in training and selection (observed_mix 0), evaluated with observed '
              'facts off. Otherwise the shipped recipe.',
    'wide': 'KG-ICL adapter refitted with wider bounds (scale bound 8 instead of 2, shift bound 16 instead of 8), since the '
            'shipped KG-ICL adapter sits at both bounds. Otherwise the shipped recipe.',
    'target-valid': "One adapter per target, fitted on 2i/3i queries of the target's released validation split and selected "
                    'on 16 validation queries per type; no test query or answer is read. Upper reference point.',
}
# UltraQuery's released weights as a frozen ULTRA checkpoint (fit.py writes it from the pinned release).
ULTRAQUERY_RELEASE = ('Experiments/query-baselines/upstream/ultra/ckpts/ultraquery.pth',
                      '9b6dc20801c35e7c9ebe65764acb4ed9cb5bffc6b0a0a727bd48935589e9388d')
ULTRAQUERY_AS_ULTRA = 'checkpoints/ultraquery_as_ultra.pth'
VARIANTS = {
    'traversal': 'Opt-in observed-fact traversal (not an upstream option): every projection also takes the maximum with an '
                 'exact traversal of the observed facts from the same fuzzy set, like the observed facts of the KGFM recipes. '
                 'Oracles use upstream SymbolicTraversal. Otherwise the released UltraQuery recipe.',
    'lp-4g': "UltraQuery LP with the frozen ULTRA 4g checkpoint, UltraQuery's initialization (upstream "
             'config/ultraquery/pretrain.yaml), and the same per-dataset thresholds.',
    'uq-weights': "UltraQuery's released weights as a frozen ULTRA backbone with the shipped adapter recipe refitted for them; "
                  'relation reasoning is conditioned on the queried relation, as in their training (relation_conditioning query).',
    'calibrated': "Opt-in calibration in UltraQuery's own executor (review follow-up of 2026-10-08, not an upstream option): "
                  'every projection replaces the sigmoid by the softmax over entities times the observed-tail mass reached '
                  'from the fuzzy set (at least one), capped at 0.9999 (QTO), and takes the maximum with the exact traversal '
                  'of the observed facts; no membership threshold. Oracles use upstream parts and SymbolicTraversal. '
                  'Otherwise the released recipe of the same weights.',
    'no-control': 'The 14-type adapters, which selection by source-validation MRR alone prefers (ULTRA 21.03 against 20.80, '
                  'TRIX 21.61 against 21.16 over the ten selection types), as PROTOCOL.md promises; no identity control, '
                  'since identity calibration does not depend on the adapter recipe and kgfm-b64 ran it.',
}
# Adapter source graphs: training split below the input root, its SHA-256, and the fingerprint of the prepared
# training data (seed 2026090851, 30% of fact pairs masked) with this relative source path. With the absolute path
# the original sweep recorded, the same data has the fingerprints in every source adapter's 'sources' metadata.
SOURCES = {
    'FB15k237': ('KGs/FB15k-237/train.txt', '6e4c2782169af21e9743f3b1d200886f5d595bf6bc504ec1351720949c5cdfae',
                 '3f311b38050ea981916aff108b6488b934e12b6a9d64d67fe57a60c5f7d57982'),
    'WN18RR': ('KGs/WN18RR/train.txt', '038612e783c215ee5f3ca9fbfca27b8d0739be1028fe4ee7c174aecf0b83d5df',
               '951b4e8c8de242e92bd109008e45b8f7d73c913f5182c84af1b2b2f87963ce3f'),
    'CoDExMedium': ('KGs/CoDEx-Medium/train.txt', 'd99c3437ab51690391a26d96976adf6e5494dba7ef6902e77000551bfa566556',
                    '7dac22366fe1d0ba509e44a9ea0279f18f273d4d829ddc2aa6d6a2e677be4759'),
}
SOURCE_URLS = {  # Public copies with the pinned bytes; FB15k-237 and WN18RR ship in DICE's KGs.zip.
    'FB15k237': 'https://files.dice-research.org/datasets/dice-embeddings/KGs.zip',
    'WN18RR': 'https://files.dice-research.org/datasets/dice-embeddings/KGs.zip',
    'CoDExMedium': 'https://raw.githubusercontent.com/tsafavi/codex/3132e426c2a6b643b70bad679905a3a6270be440/data/triples/codex-m/train.txt',
}


@dataclass(frozen=True)
class Study:
    """One frozen final study.

    Attributes:
        suite: ``ultraquery`` (UQ-23) or ``plus_h``.
        name: Study name; ``suite-name`` is its directory below the output root.
        recipes: Recipe files of ``benchmarks/<suite>``, in the order ``prepare`` read them.
        entries: Number of frozen entries.
        hours: Measured test cost: the sum of every entry's ``total_wall_seconds``, in hours, on H100 NVL GPUs
            with up to two jobs per GPU.
        include: Entry-ID prefixes to keep (default: every entry).
        exclude: Entry-ID prefixes to drop.
        bracket: ``global``, ``uqlp`` or ``target-fit``: the same recipes with other adapter weights and no paired
            identity control, as registered on 2026-10-05; or a review reference point of ``REVIEW`` (2026-10-08).
        variant: A review variant of ``VARIANTS`` (2026-10-08) that changes the selected recipes' method settings.
        observed_facts: ``prepare --observed-facts`` mode.
        hardware_profile: ``prepare --hardware-profile``.
        difficulty: Whether the paper uses this +H study's answer-difficulty report.
        superseded: Entry-ID prefixes of frozen entries that were stopped and are reported from another study.
        summary: One line for listings.
    """

    suite: str
    name: str
    recipes: tuple[str, ...]
    entries: int
    hours: float
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    bracket: str | None = None
    observed_facts: str | None = None
    hardware_profile: str = 'h100'
    difficulty: bool = False
    superseded: tuple[str, ...] = ()
    summary: str = ''
    variant: str | None = None

    @property
    def id(self) -> str:
        """The study's directory name, ``suite-name``."""
        return f'{self.suite}-{self.name}'


def _suite(suite: str, counts: dict, hours: dict) -> tuple[Study, ...]:
    """The final studies of one suite; UQ-23 adds the UltraQuery LP thresholds and the per-graph baselines."""
    kgfm = ('kgfm_adapters.json',)
    kgicl = dict(recipes=('kgfm_kgicl.json',), include=KGICL, exclude=KGICL_SEEDS)
    plus_h = suite == 'plus_h'  # The paper's hardness figure and appendix bin these +H studies by answer difficulty.
    specs: list[tuple[str, dict]] = [
        ('kgfm-b64', dict(recipes=(*kgfm, 'kgfm_seeds.json'), difficulty=plus_h,
                          summary='ULTRA and TRIX, five adapter training seeds; seed 0 with its identity control')),
        ('kgfm-b64-ablations', dict(recipes=('kgfm_ablations.json',), difficulty=plus_h,
                                    summary='Adapter fitted without FB15k237; ULTRA 4g and 50g backbones')),
        ('kgfm-b64-facts-none', dict(recipes=kgfm, observed_facts='none', summary='Observed facts off, seed-0 adapters and controls')),
        ('kgfm-b64-global', dict(recipes=kgfm, bracket='global', summary='Global 2-parameter calibration (lower reference)')),
        *([] if plus_h else [('kgfm-b64-uqlp', dict(recipes=kgfm, bracket='uqlp', summary="UltraQuery LP's thresholds (lower reference)"))]),
        ('kgfm-b64-target-fit', dict(recipes=kgfm, bracket='target-fit', summary='One adapter fitted per target graph (upper reference)')),
        ('kgfm-b64-kgicl', dict(kgicl, difficulty=plus_h, summary='KG-ICL with its adapter and identity control')),
        ('kgfm-b64-kgicl-seeds', dict(recipes=('kgfm_kgicl.json',), include=KGICL_SEEDS, summary='KG-ICL adapter seeds 1-4')),
        ('kgfm-b64-kgicl-facts-none', dict(kgicl, observed_facts='none', summary='KG-ICL with observed facts off')),
        ('kgfm-b64-kgicl-global', dict(kgicl, bracket='global', summary='KG-ICL global 2-parameter calibration')),
        # QTO's entries of the table-baseline studies were stopped after 1p (2026-10-04 23:15); the qto study reports them.
        ('baselines', dict(recipes=('baselines.json',), hardware_profile='default', exclude=('qto-',), difficulty=plus_h,
                           superseded=('qto-',) if plus_h else (),
                           summary='The seven published +H methods' if plus_h else 'UltraQuery port with released weights')),
        *([] if plus_h else [('pergraph', dict(recipes=('trained_baselines.json', 'comparisons.json'), hardware_profile='default',
                                               exclude=('qto-', 'ultraquery-lp-no-threshold-'), superseded=('qto-',),
                                               summary='GNN-QE (inductive e), incoming-relation heuristic, UltraQuery LP'))]),
        ('qto', dict(recipes=('baselines.json',) if plus_h else ('comparisons.json', 'trained_baselines.json'),
                     hardware_profile='default', include=('qto-',), difficulty=plus_h, summary='QTO with relation matrices built once')),
    ]
    return tuple(Study(suite, name, entries=counts[name], hours=hours[name], **settings) for name, settings in specs)


def _review_suite(suite: str) -> tuple[Study, ...]:
    """The review experiments of one suite (registered 2026-10-08); entries and hours are filled in once they ran."""
    kgfm = ('kgfm_adapters.json',)
    kgicl = dict(recipes=('kgfm_kgicl.json',), include=KGICL, exclude=KGICL_SEEDS)
    plus_h = suite == 'plus_h'
    specs: list[tuple[str, dict]] = [
        ('kgfm-b64-minmax', dict(recipes=kgfm, bracket='minmax', difficulty=plus_h, summary="CQD-Hybrid's min-max calibration (training-free)")),
        ('kgfm-b64-softmax-degree', dict(recipes=kgfm, bracket='softmax-degree', difficulty=plus_h,
                                         summary="QTO's softmax-times-degree calibration (training-free)")),
        ('kgfm-b64-kgicl-minmax', dict(kgicl, bracket='minmax', difficulty=plus_h, summary='KG-ICL with min-max calibration')),
        ('kgfm-b64-kgicl-softmax-degree', dict(kgicl, bracket='softmax-degree', difficulty=plus_h,
                                               summary='KG-ICL with softmax-times-degree calibration')),
        ('kgfm-b64-softmax-degree-facts-none', dict(recipes=kgfm, include=('ultra-',), bracket='softmax-degree', observed_facts='none',
                                                    difficulty=plus_h, summary="QTO's softmax-times-degree calibration without "
                                                    'the known-fact override (ULTRA; review follow-up of 2026-10-08)')),
        ('kgfm-b64-gamma0', dict(recipes=kgfm, bracket='gamma0', observed_facts='none', difficulty=plus_h,
                                 summary='Adapters refitted and evaluated without observed facts')),
        ('kgfm-b64-kgicl-gamma0', dict(kgicl, bracket='gamma0', observed_facts='none', difficulty=plus_h,
                                       summary='KG-ICL adapter refitted and evaluated without observed facts')),
        ('kgfm-b64-kgicl-wide', dict(kgicl, bracket='wide', difficulty=plus_h, summary='KG-ICL adapter with wider bounds')),
        ('kgfm-b64-uqweights', dict(recipes=kgfm, include=('ultra-',), variant='uq-weights', difficulty=plus_h,
                                    summary="UltraQuery's weights as a frozen backbone with a refitted adapter and identity control")),
        ('kgfm-b64-target-valid', dict(recipes=kgfm, bracket='target-valid', difficulty=plus_h,
                                       summary='One adapter fitted per target on its validation queries (upper reference)')),
        ('kgfm-b64-types14', dict(recipes=('kgfm_14types.json',), variant='no-control', difficulty=plus_h,
                                  summary='14-type adapters, preferred by source-validation MRR alone')),
        ('baselines-traversal', dict(recipes=('baselines.json',), include=('ultraquery-',), variant='traversal',
                                     hardware_profile='default', difficulty=plus_h, summary='UltraQuery with observed-fact traversal')),
        ('baselines-calibrated', dict(recipes=('baselines.json',), include=('ultraquery-',), variant='calibrated',
                                      hardware_profile='default', difficulty=plus_h,
                                      summary='UltraQuery with known facts and softmax-times-degree projections')),
        *([] if plus_h else [('pergraph-lp-calibrated', dict(recipes=('comparisons.json',), include=('ultraquery-lp-',),
                                                             exclude=('ultraquery-lp-no-threshold-',), variant='calibrated',
                                                             hardware_profile='default',
                                                             summary="Frozen ULTRA 3g in UltraQuery's executor with known facts "
                                                                     'and softmax-times-degree projections'))]),
        *([] if plus_h else [('pergraph-lp-4g', dict(recipes=('comparisons.json',), include=('ultraquery-lp-',),
                                                     exclude=('ultraquery-lp-no-threshold-',), variant='lp-4g', hardware_profile='default',
                                                     summary='UltraQuery LP with the frozen ULTRA 4g checkpoint'))]),
    ]
    return tuple(Study(suite, name, entries=0, hours=0., **settings) for name, settings in specs)


STUDIES = (
    *_suite('ultraquery',
            counts={'kgfm-b64': 230, 'kgfm-b64-ablations': 92, 'kgfm-b64-facts-none': 46, 'kgfm-b64-global': 46, 'kgfm-b64-uqlp': 46,
                    'kgfm-b64-target-fit': 46, 'kgfm-b64-kgicl': 23, 'kgfm-b64-kgicl-seeds': 92, 'kgfm-b64-kgicl-facts-none': 23,
                    'kgfm-b64-kgicl-global': 23, 'baselines': 23, 'pergraph': 43, 'qto': 3},
            hours={'kgfm-b64': 140.6, 'kgfm-b64-ablations': 44.9, 'kgfm-b64-facts-none': 42.3, 'kgfm-b64-global': 24.2,
                   'kgfm-b64-uqlp': 17.8, 'kgfm-b64-target-fit': 24.7, 'kgfm-b64-kgicl': 21.1, 'kgfm-b64-kgicl-seeds': 49.9,
                   'kgfm-b64-kgicl-facts-none': 21.8, 'kgfm-b64-kgicl-global': 12.0, 'baselines': 5.1, 'pergraph': 9.2, 'qto': 0.7}),
    *_suite('plus_h',
            counts={'kgfm-b64': 30, 'kgfm-b64-ablations': 12, 'kgfm-b64-facts-none': 6, 'kgfm-b64-global': 6, 'kgfm-b64-target-fit': 6,
                    'kgfm-b64-kgicl': 3, 'kgfm-b64-kgicl-seeds': 12, 'kgfm-b64-kgicl-facts-none': 3, 'kgfm-b64-kgicl-global': 3,
                    'baselines': 18, 'qto': 3},
            hours={'kgfm-b64': 18.1, 'kgfm-b64-ablations': 5.7, 'kgfm-b64-facts-none': 5.2, 'kgfm-b64-global': 3.3,
                   'kgfm-b64-target-fit': 3.5, 'kgfm-b64-kgicl': 1.6, 'kgfm-b64-kgicl-seeds': 3.7, 'kgfm-b64-kgicl-facts-none': 1.9,
                   'kgfm-b64-kgicl-global': 1.0, 'baselines': 8.9, 'qto': 1.3}),
)


# Review experiments (2026-10-08): selectable by name, not part of --all until their results are final.
REVIEW_STUDIES = (*_review_suite('ultraquery'), *_review_suite('plus_h'))


def select_studies(patterns: list[str] | None) -> list[Study]:
    """Studies matching ``patterns``, in registry order.

    Args:
        patterns: Study IDs (``plus_h-kgfm-b64``), names matched in every suite (``kgfm-b64``) or shell patterns
            (``'plus_h-*'``); ``None`` selects every study.

    Raises:
        ValueError: If a pattern matches no study.
    """
    if not patterns:
        return list(STUDIES)
    chosen = set()
    for pattern in patterns:
        def match(studies):
            return {s.id for s in studies if fnmatch.fnmatchcase(s.id, pattern) or fnmatch.fnmatchcase(s.name, pattern)}

        # A pattern selects final studies; review studies only when it matches no final one.
        matches = match(STUDIES) or match(REVIEW_STUDIES)
        if not matches:
            hint = 'pass --all for every study' if pattern == 'all' else 'see python -m benchmarks.cqa reproduce --list'
            raise ValueError(f'No study matches {pattern!r}; {hint}')
        chosen |= matches
    return [study for study in (*STUDIES, *REVIEW_STUDIES) if study.id in chosen]


@cache
def _thresholds() -> tuple[tuple[str, float], ...]:
    entries = read_manifest(suite_directory('ultraquery') / 'comparisons.json')['entries']
    return tuple((e['dataset'], e['options']['threshold']) for e in entries if e['method'] == 'ultraquery-lp' and e['options'].get('threshold'))


def thresholds() -> dict[str, float]:
    """UltraQuery LP's membership threshold per UQ-23 dataset, from its public recipe."""
    values = dict(_thresholds())
    if len(values) != len(catalog().BENCHMARK_DATASETS):
        raise ValueError(f'Expected a threshold for every UQ-23 dataset, found {len(values)}')
    return values


def bracket_path(kind: str, backbone: str, dataset: str) -> str:
    """Adapter of a calibration bracket or review reference point, relative to the adapter root."""
    if kind == 'global':
        return f'brackets/{backbone}/global.json'
    if kind == 'uqlp':
        return f'brackets/{backbone}/uqlp-threshold-{thresholds()[dataset]}.json'
    if kind == 'target-fit':
        return f'brackets/{backbone}/target-fit/{dataset.replace(":", "-")}.json'
    if kind == 'target-valid':
        return f'review/{backbone}/target-valid/{dataset.replace(":", "-")}.json'
    if kind not in REVIEW:
        raise ValueError(f'Unknown bracket {kind!r}')
    return f'review/{backbone}/{kind}.json'


def _variant(entry: dict, variant: str) -> dict:
    """``entry`` with the method settings of a review variant, under its own ID."""
    reference = dict(status=REVIEW_STATUS, notes=[VARIANTS[variant], 'Every other setting equals the selected recipe.'])
    if variant == 'traversal':
        return dict(entry, id=entry['id'].replace('ultraquery-', 'ultraquery-traversal-', 1), reference=reference,
                    options=dict(entry['options'], observed_traversal=True))
    if variant == 'no-control':
        return dict(entry, reference=reference, adapter_ablation=False)
    if variant == 'calibrated':
        prefix = 'ultraquery-lp-' if entry['id'].startswith('ultraquery-lp-') else 'ultraquery-'
        return dict(entry, id=entry['id'].replace(prefix, f'{prefix}calibrated-', 1), reference=reference,
                    options=dict(entry['options'], observed_traversal=True, calibration='softmax-degree', threshold=0.))
    if variant == 'lp-4g':
        return dict(entry, id=entry['id'].replace('ultraquery-lp-', 'ultraquery-lp-4g-', 1), reference=reference,
                    checkpoint='checkpoints/ultra_4g.pth')
    return dict(entry, id=f'ultra-uqweights-{entry["dataset"]}', reference=reference, checkpoint=ULTRAQUERY_AS_ULTRA,
                options=dict(entry['options'], relation_conditioning='query'), adapter_ablation=True, finalized=True, blockers=[],
                adapters={'product': f'{ADAPTER_ROOT}/review/ultraquery-weights/product_intersections.json'})


def recipe_manifest(study: Study, adapter_root: str = ADAPTER_ROOT) -> dict:
    """The recipe manifest ``prepare`` reads for ``study``: selected entries, bracket adapters, adapters below ``adapter_root``.

    Args:
        study: A registered study.
        adapter_root: Directory of the adapters, relative to the input root; ``benchmarks/adapters`` holds the shipped
            ones, which every final study used.
    """
    sources = [read_manifest(suite_directory(study.suite) / name) for name in study.recipes]
    manifest = {key: value for key, value in sources[0].items() if key != 'entries'}
    entries = [entry for source in sources for entry in source['entries']
               if (not study.include or entry['id'].startswith(study.include)) and not entry['id'].startswith(study.exclude)]
    if study.variant:
        entries = [_variant(entry, study.variant) for entry in entries]
    if study.bracket:
        review = study.bracket in REVIEW
        reference = dict(status=REVIEW_STATUS if review else BRACKET_STATUS,
                         notes=[(REVIEW if review else BRACKETS)[study.bracket], 'Every other setting equals the shipped recipe of this backbone.'])
        entries = [dict(entry, id=f'{entry["method"].removesuffix("-adapter")}-{study.bracket}-{entry["dataset"]}', adapter_ablation=False,
                        reference=copy.deepcopy(reference), finalized=True, blockers=[],
                        adapters={'product': f'{ADAPTER_ROOT}/{bracket_path(study.bracket, entry["method"].removesuffix("-adapter"), entry["dataset"])}'})
                   for entry in entries]
    prefix = ADAPTER_ROOT + '/'
    for entry in entries:
        if 'adapters' in entry:
            entry['adapters'] = {name: f'{adapter_root}/{path.removeprefix(prefix)}' if path.startswith(prefix) else path
                                 for name, path in entry['adapters'].items()}
    return dict(manifest, entries=entries)


def prepare_options(study: Study) -> list[str]:
    """``prepare`` flags of ``study`` besides its recipe manifest."""
    options = ['--hardware-profile', study.hardware_profile] if study.hardware_profile != 'default' else []
    return options + (['--observed-facts', study.observed_facts] if study.observed_facts else [])


def resolved_manifest(study: Study, adapter_root: str = ADAPTER_ROOT) -> dict:
    """The manifest ``prepare`` freezes for ``study`` (its ``manifest.json``), resolved without inputs or PyTorch."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'recipes.json'
        path.write_text(json.dumps(recipe_manifest(study, adapter_root)))
        manifest: dict = prepare_manifest([path], suite=study.suite, answer_filter=SUITES[study.suite]['answer_filter'],
                                hardware_profile=study.hardware_profile,
                                observed_facts=[study.observed_facts] if study.observed_facts else None)
    return manifest


@dataclass(frozen=True)
class Fit:
    """How one adapter below the adapter root was made (2026-09-25 to 2026-10-06).

    ``source`` fits train on 2i/3i queries of the masked FB15k237, WN18RR and CoDExMedium training graphs and select
    the checkpoint on their validation queries; refitting the reference seed reproduced the shipped ULTRA and TRIX
    adapters to within 2e-6 (backbone scores are recomputed on the GPU). ``target`` fits use queries sampled from one
    target's test inference graph instead; ``threshold`` adapters are built, not trained.
    """

    backbone: str
    kind: str = 'source'
    seed: int = SEED
    feature_mode: str = 'context_scores_v1'
    checkpoint: str | None = None
    without: tuple[str, ...] = ()
    target: str | None = None
    threshold: float | None = None
    observed_mix: float = 1.
    bias_bound: float = 8.
    scale_bound: float = 2.
    relation_conditioning: str = 'direct'
    calibration: str | None = None
    mask_known: bool = False
    sources: str | None = None  # prepared source queries below the input root instead of the regenerated standard ones

    @property
    def group(self) -> str:
        """Fits of one group share score banks, so they run one after another in one worker."""
        if self.kind == 'source':
            return self.backbone + (f'-{Path(self.checkpoint).stem}' if self.checkpoint else '')
        if self.kind in ('target', 'target-valid'):
            return f'{self.backbone}-{self.kind}-{str(self.target).replace(":", "-")}'
        return f'{self.backbone}-thresholds' if self.kind == 'threshold' else f'{self.backbone}-calibrations'


def _fits() -> dict[str, Fit]:
    fits = {f'{backbone}_product_intersections.json': Fit(backbone) for backbone in BACKBONES}
    fits |= {f'seeds/{backbone}_product_intersections_seed{seed}.json': Fit(backbone, seed=seed) for backbone in BACKBONES for seed in (1, 2, 3, 4)}
    fits |= {f'ablations/{backbone}_product_intersections_no_fb15k237.json': Fit(backbone, without=('FB15k237',)) for backbone in ('ultra', 'trix')}
    fits |= {f'ablations/ultra_{name}_product_intersections.json': Fit('ultra', checkpoint=f'checkpoints/ultra_{name}.pth') for name in ('4g', '50g')}
    fits |= {bracket_path('global', backbone, ''): Fit(backbone, feature_mode='global') for backbone in BACKBONES}
    for backbone in ('ultra', 'trix'):
        fits |= {f'brackets/{backbone}/uqlp-threshold-{value}.json': Fit(backbone, kind='threshold', threshold=value)
                 for value in sorted(set(thresholds().values()))}
        fits |= {bracket_path('target-fit', backbone, dataset): Fit(backbone, kind='target', target=dataset)
                 for dataset in (*catalog().BENCHMARK_DATASETS, *catalog().PLUS_H_DATASETS)}
    return fits


def _review_fits() -> dict[str, Fit]:
    """Adapters of the review experiments (2026-10-08)."""
    fits = {}
    for backbone in BACKBONES:
        fits |= {bracket_path(name, backbone, ''): Fit(backbone, kind='calibration', calibration=name) for name in ('minmax', 'softmax-degree')}
        fits[bracket_path('gamma0', backbone, '')] = Fit(backbone, observed_mix=0.)
        fits |= {bracket_path('target-valid', backbone, dataset): Fit(backbone, kind='target-valid', target=dataset)
                 for dataset in (*catalog().BENCHMARK_DATASETS, *catalog().PLUS_H_DATASETS)}
    fits[bracket_path('wide', 'kgicl', '')] = Fit('kgicl', scale_bound=8., bias_bound=16.)
    fits['review/ultraquery-weights/product_intersections.json'] = Fit('ultra', checkpoint=ULTRAQUERY_AS_ULTRA, relation_conditioning='query')
    # Flock (one prediction per atom, 128 walks), fitted like the shipped recipe; evaluated on a query subset.
    fits['review/flock/product_intersections.json'] = Fit('flock')
    # Second review round (2026-10-08 20:36): training-free variants for the subset arms, each bound to its weights.
    fits['review/ultra/softmax.json'] = Fit('ultra', kind='calibration', calibration='softmax')
    fits['review/kgicl/softmax.json'] = Fit('kgicl', kind='calibration', calibration='softmax')
    fits['review/kgicl/softmax-degree-ties.json'] = Fit('kgicl', kind='calibration', calibration='softmax-degree-ties')
    fits['review/trix/softmax-degree-ties.json'] = Fit('trix', kind='calibration', calibration='softmax-degree-ties')
    fits['review/trix/softmax-degree-ties-masked.json'] = Fit('trix', kind='calibration', calibration='softmax-degree-ties',
                                                              mask_known=True)
    fits['review/flock/softmax-degree.json'] = Fit('flock', kind='calibration', calibration='softmax-degree')
    fits['review/flock/softmax.json'] = Fit('flock', kind='calibration', calibration='softmax')
    fits['review/ultra-4g/softmax-degree.json'] = Fit('ultra', kind='calibration', calibration='softmax-degree',
                                                      checkpoint='checkpoints/ultra_4g.pth')
    fits['review/ultraquery-weights/softmax-degree.json'] = Fit('ultra', kind='calibration', calibration='softmax-degree',
                                                                checkpoint=ULTRAQUERY_AS_ULTRA, relation_conditioning='query')
    # Generator bias: KG-ICL's five adapter seeds and its global calibration refitted on hardness-balanced source queries
    # (screening candidate 4's generator: same graphs, masks, types and counts, hardness levels in rotation).
    balanced = 'Experiments/review/sources-balanced'
    fits |= {f'review/kgicl/hb/seed{seed}.json': Fit('kgicl', seed=seed, sources=balanced) for seed in (SEED, 1, 2, 3, 4)}
    fits['review/kgicl/hb/global.json'] = Fit('kgicl', feature_mode='global', sources=balanced)
    return fits


FITS = _fits()
REVIEW_FITS = _review_fits()
ALL_FITS = FITS | REVIEW_FITS


def study_adapters(study: Study) -> list[str]:
    """Adapters ``study`` reads, relative to the adapter root, in first-use order."""
    prefix = ADAPTER_ROOT + '/'
    paths = [path.removeprefix(prefix) for entry in resolved_manifest(study)['entries'] for path in entry.get('adapters', {}).values()]
    return list(dict.fromkeys(paths))


def sha256(path: str | Path) -> str:
    """SHA-256 of a file."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def scientific_entry(entry: dict, adapter_sha256) -> dict:
    """``entry`` without its documentation (``reference``) and with adapters named by content, for equivalence checks.

    Args:
        entry: A resolved manifest entry.
        adapter_sha256: Maps an adapter path of the entry to the SHA-256 of its bytes.
    """
    result = {key: value for key, value in entry.items() if key != 'reference'}
    if 'adapters' in entry:
        result['adapters'] = {name: adapter_sha256(path) for name, path in entry['adapters'].items()}
    return result


def compare_manifest(study: Study, frozen: dict, frozen_sha256, regenerated: dict | None = None) -> dict:
    """Compare a frozen study manifest with the one the registry resolves.

    Args:
        study: The registered study.
        frozen: The study's frozen ``manifest.json``.
        frozen_sha256: Maps an adapter path of the frozen manifest to the SHA-256 of the adapter it named.
        regenerated: The resolved manifest (default: ``resolved_manifest(study)`` with the shipped adapters).

    Returns:
        ``scientific`` (whether every field but the documentation and adapter paths is identical, adapters by content),
        ``superseded`` (frozen entries left out by design), and the IDs of entries whose adapter paths or
        ``reference`` documentation differ.
    """
    regenerated = regenerated if regenerated is not None else resolved_manifest(study)
    kept = [entry for entry in frozen['entries'] if not entry['id'].startswith(study.superseded)]
    superseded = [entry['id'] for entry in frozen['entries'] if entry['id'].startswith(study.superseded)]

    def top(manifest):
        return {key: value for key, value in manifest.items() if key != 'entries'}

    def ours(path):
        return sha256(REPO / path)

    same = (top(frozen) == top(regenerated) and [e['id'] for e in kept] == [e['id'] for e in regenerated['entries']]
            and all(scientific_entry(a, frozen_sha256) == scientific_entry(b, ours) for a, b in zip(kept, regenerated['entries'])))
    return dict(study=study.id, scientific=same, entries=len(regenerated['entries']), superseded=superseded,
                adapter_paths=[a['id'] for a, b in zip(kept, regenerated['entries']) if a.get('adapters') != b.get('adapters')],
                reference=[a['id'] for a, b in zip(kept, regenerated['entries']) if a.get('reference') != b.get('reference')])
