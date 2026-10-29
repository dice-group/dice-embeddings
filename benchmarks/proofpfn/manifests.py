"""Frozen graph and model manifests of the ProofPFN benchmark, resolved without PyTorch.

A graph manifest (``manifests/<name>.json``) lists graphs with their folders, their audit fields and the hashes that
each graph must match before it loads:

- ``dev121``: the 121 audited development graphs of ``graph-audit-v1``. Views: ``main`` (the 91 graphs of the groups
  E, L, B, S, O and R, with their weights), ``contamination`` (29) and ``apart`` (UMLS). The audit recorded the content
  hashes of ``train.txt`` and ``test.txt`` (SHA-256 of the sorted distinct triples); the file hashes of a folder are
  listed where the standard suite manifest records the same folder, and ``null`` elsewhere. Test sample: the
  ``first-1000`` rule.
- ``t1``: the 54 zero-shot graphs of the standard suite ``p0-suite-v1`` with their suite ids (the keys of Flock's walk
  table), file hashes and counts. Views: the suite groups ``trans``, ``ind_e`` and ``ind_er``. Every test row is ranked.
- ``gate``: the three graphs of the exact-rank goldens (UMLS, ``ultra-suite-v1/NELLInductive_v1``,
  ``inductive/grail-WN18RR_v1``), with file and content hashes. The audit excludes grail-WN18RR_v1 as a copy of
  ``ultra-suite-v1/WN18RRInductive_v1``, so it is not a graph of ``dev121``; the gate lists it explicitly.

Blind sets. A manifest with ``"blind": true`` (a confirmatory test set) is opened only with a release flag that names
it, and only a blind manifest may name a folder below ``BLIND_FOLDERS``. This branch holds no blind manifest.

``models.json`` pins the research exports of the ProofPFN families (seeds 42, 43 and 44) and the reference checkpoints.
HT's pins come from the research ``models-pcuht.json``, written after HT's training. A family whose pins are ``null``
is refused under its registered key.

The ``build_*`` functions derive the graph manifests from the research inputs named in each manifest's ``sources``;
a test rebuilds them when the inputs are available.
"""
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = Path(__file__).resolve().parent / 'manifests'
FORMAT = 'proofpfn-graph-manifest-v1'
MODELS_FORMAT = 'proofpfn-models-v1'
SPLITS = ('train', 'valid', 'test')
CONTENT_SPLITS = ('train', 'test')
SEEDS = (42, 43, 44)
BLIND_FOLDERS = ('KGs/fresh-v1', 'KGs/lockbox3')
SAMPLE, SAMPLE_SEED = 1000, 0
SAMPLE_RULES = {
    'first-1000': ('test rows <= 1000: all rows (fair_eval); else the first occurrence of every distinct test fact in file '
                   'order, sorted(random.Random(0).sample(pool, 1000)); index = position in test.txt'),
    'all': 'every test row; index = position in test.txt',
}
GROUP_CODE = {'E': 'E', 'L': 'L', 'B': 'B', 'S': 'S', 'O': 'O', 'R': 'R', 'T (reported apart)': 'T', 'contamination view': 'C'}
VIEW_OF_CODE = {**dict.fromkeys('ELBSOR', 'main'), 'C': 'contamination', 'T': 'apart'}
GATE_GRAPHS = ('UMLS', 'ultra-suite-v1/NELLInductive_v1', 'inductive/grail-WN18RR_v1')
PathLike = Union[str, Path]


class ManifestError(ValueError):
    """A manifest that is malformed, or a graph that fails its hash check."""


class BlindSetError(PermissionError):
    """A blind test set opened without its release flag, or a blind folder named by a development manifest."""


@dataclass(frozen=True)
class GraphEntry:
    """One graph of a manifest.

    Attributes:
        key: The graph's key (``<collection folder>/<name>`` for most graphs); row files record it as ``allgraph.key``.
        name: The graph name, the stem of its row file.
        path: The folder, relative to the data root.
        collection: The collection of the row folder ``<out>/<collection>/<systems>/<name>.json``.
        view: The view that holds the graph (``main``, ``contamination``, ``apart``, a suite group, or ``gate``).
        code: The audit group code (E, L, B, S, O, R, C, T), or ``None``.
        group: The audit group as written in the audit, or the suite group.
        verdict: The audit verdict (``keep``, ``keep-flag``, ``exclude``), or ``None``.
        weight: The graph's weight in the main view (zero elsewhere).
        origin_class: ``published`` or ``converted``, or ``None``.
        suite_id: The standard-suite id (the key of Flock's walk table), or ``None``.
        size: ``entities`` and ``inference_edges`` of ``train.txt`` (distinct entities, distinct triples).
        sha256: The file SHA-256 per split, ``None`` where no manifest input records it.
        content_sha256: The audit's content hash per split (``train``, ``test``), where recorded.
        counts: The suite manifest's counts, or ``None``.
        tail_only_view: The suite's tail-only flag, or ``None``.
    """

    key: str
    name: str
    path: str
    collection: str
    view: str
    code: Optional[str]
    group: Optional[str]
    verdict: Optional[str]
    weight: float
    origin_class: Optional[str]
    suite_id: Optional[str]
    size: Mapping[str, int]
    sha256: Mapping[str, Optional[str]]
    content_sha256: Mapping[str, str]
    counts: Optional[Mapping[str, int]]
    tail_only_view: Optional[bool]

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> 'GraphEntry':
        """An entry from its manifest JSON object.

        Raises:
            ManifestError: When a field is missing or malformed.
        """
        try:
            entry = cls(key=value['key'], name=value['name'], path=value['path'], collection=value['collection'], view=value['view'],
                        code=value.get('code'), group=value.get('group'), verdict=value.get('verdict'), weight=float(value.get('weight', 0.0)),
                        origin_class=value.get('origin_class'), suite_id=value.get('suite_id'),
                        size=MappingProxyType(dict(value['size'])), sha256=MappingProxyType({s: value['sha256'].get(s) for s in SPLITS}),
                        content_sha256=MappingProxyType(dict(value.get('content_sha256') or {})),
                        counts=MappingProxyType(dict(value['counts'])) if value.get('counts') is not None else None,
                        tail_only_view=value.get('tail_only_view'))
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            raise ManifestError(f'malformed graph entry {value.get("key") if isinstance(value, Mapping) else value!r}: {error!r}') from None
        if Path(entry.path).is_absolute() or '..' in Path(entry.path).parts or not entry.name or '/' in entry.name:
            raise ManifestError(f'{entry.key}: the path must be relative to the data root and the name a plain file stem')
        if set(entry.content_sha256) - set(CONTENT_SPLITS):
            raise ManifestError(f'{entry.key}: content hashes exist for train and test only')
        return entry

    def record(self) -> dict[str, Any]:
        """The entry's fields for the ``allgraph`` record of a row file (with the suite's tail-only flag where it has one)."""
        record = dict(key=self.key, collection=self.collection, path=self.path, group=self.group, verdict=self.verdict, weight=self.weight,
                      origin_class=self.origin_class, suite_id=self.suite_id)
        if self.tail_only_view is not None:
            record['tail_only_view'] = self.tail_only_view
        return record


@dataclass(frozen=True)
class Manifest:
    """A frozen graph set.

    Attributes:
        name: The manifest name (``dev121``, ``t1``, ``gate``).
        about: What the set is and where it comes from.
        blind: True for a confirmatory test set, which opens only with its release flag.
        sample_rule: ``first-1000`` or ``all`` (``SAMPLE_RULES``).
        sources: The input files of the build and their SHA-256.
        entries: The graphs, in manifest order.
        sha256: The SHA-256 of the manifest file.
    """

    name: str
    about: str
    blind: bool
    sample_rule: str
    sources: Mapping[str, str]
    entries: tuple[GraphEntry, ...]
    sha256: str

    def __post_init__(self) -> None:
        names, keys = [e.name for e in self.entries], [e.key for e in self.entries]
        if len(set(names)) != len(names) or len(set(keys)) != len(keys):
            raise ManifestError(f'{self.name}: graph names and keys must be unique')
        if self.sample_rule not in SAMPLE_RULES:
            raise ManifestError(f'{self.name}: unknown sample rule {self.sample_rule!r}')
        if not self.blind:
            hidden = [e.path for e in self.entries if in_blind_folder(e.path)]
            if hidden:
                raise BlindSetError(f'{self.name} is not a blind manifest but names blind folders: {hidden[:3]}')

    @property
    def views(self) -> dict[str, tuple[str, ...]]:
        """View name to graph names, with ``all``."""
        out: dict[str, list[str]] = {}
        for entry in self.entries:
            out.setdefault(entry.view, []).append(entry.name)
        return {'all': tuple(e.name for e in self.entries), **{k: tuple(v) for k, v in out.items()}}

    def entry(self, graph: str) -> GraphEntry:
        """The entry of a graph name or key.

        Raises:
            KeyError: For a graph outside the manifest.
        """
        for entry in self.entries:
            if graph in (entry.name, entry.key):
                return entry
        raise KeyError(f'{graph!r} is not a graph of manifest {self.name}')

    def select(self, view: Optional[str] = None, graphs: Optional[Iterable[str]] = None, codes: Optional[Iterable[str]] = None,
               verdicts: Optional[Iterable[str]] = None, collections: Optional[Iterable[str]] = None) -> list[GraphEntry]:
        """The entries of explicit graphs (names or keys, in the given order) or of a view, then filtered; no duplicates.

        Raises:
            KeyError: For an unknown graph or view.
            ValueError: When neither graphs nor a view is given.
        """
        if graphs:
            wanted = [self.entry(g) for g in graphs]
            if view:
                wanted = [e for e in wanted if e.name in self.views[view]]
        elif view:
            if view not in self.views:
                raise KeyError(f'{view!r} is not a view of {self.name}: {sorted(self.views)}')
            wanted = [self.entry(g) for g in self.views[view]]
        else:
            raise ValueError('choose graphs or a view')
        for attribute, allowed in (('code', codes), ('verdict', verdicts), ('collection', collections)):
            if allowed is not None:
                allowed = set(allowed)
                wanted = [e for e in wanted if getattr(e, attribute) in allowed]
        return list({e.key: e for e in wanted}.values())

    def folder(self, entry: GraphEntry, root: PathLike) -> Path:
        """The folder of an entry below the data root (a blind folder only for a blind manifest).

        Raises:
            BlindSetError: For a blind folder named by a development manifest.
        """
        if in_blind_folder(entry.path) and not self.blind:
            raise BlindSetError(f'{entry.key}: {entry.path} is a blind folder')
        return Path(root) / entry.path


def in_blind_folder(path: str) -> bool:
    """Whether a path relative to the data root lies in a blind folder (``BLIND_FOLDERS``)."""
    parts = Path(path).parts
    return any(parts[:len(Path(b).parts)] == Path(b).parts for b in BLIND_FOLDERS)


def read_manifest(source: PathLike, release: Optional[str] = None) -> Manifest:
    """Open a graph manifest by name (``manifests/<name>.json``) or path.

    Args:
        source: A manifest name or a JSON file.
        release: The name of the blind manifest that this run may open; ``None`` refuses every blind manifest.

    Returns:
        The manifest.

    Raises:
        BlindSetError: For a blind manifest whose name is not ``release``.
        ManifestError: For a malformed manifest.
    """
    path = Path(source)
    if not path.suffix and not path.exists():
        path = MANIFESTS / f'{source}.json'
    raw = path.read_bytes()
    doc = json.loads(raw)
    if not isinstance(doc, dict) or doc.get('format') != FORMAT:
        raise ManifestError(f'{path} is not a {FORMAT} file')
    blind = bool(doc.get('blind', False))
    if blind and release != doc.get('name'):
        raise BlindSetError(f"{doc.get('name')} is a blind test set; it opens only with the release flag naming it")
    return Manifest(name=doc['name'], about=doc.get('about', ''), blind=blind, sample_rule=doc['sample_rule'],
                    sources=MappingProxyType(dict(doc.get('sources', {}))),
                    entries=tuple(GraphEntry.from_json(g) for g in doc['graphs']), sha256=hashlib.sha256(raw).hexdigest())


# --------------------------------------------------------------------------------------------------------- hashes

def file_sha256(path: PathLike) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(path: PathLike) -> str:
    """The audit's content hash of a triple file: SHA-256 of its sorted distinct triples, tab-joined, one per line.

    A line splits at tabs if it has one, else at whitespace; lines without exactly three fields are skipped (the rule of
    ``graph-audit-v1``, restated by ``allgraph-v1/ag_eval.content_sha``).
    """
    rows = set()
    with Path(path).open(encoding='utf-8', errors='replace') as handle:
        for line in handle:
            line = line.rstrip('\n').rstrip('\r')
            if not line.strip():
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) == 3:
                rows.add('\t'.join(p.strip() for p in parts))
    digest = hashlib.sha256()
    for row in sorted(rows):
        digest.update(row.encode('utf-8', 'replace'))
        digest.update(b'\n')
    return digest.hexdigest()


def graph_problems(entry: GraphEntry, folder: PathLike) -> tuple[list[str], dict[str, str]]:
    """Check a graph folder against its entry before it loads.

    Args:
        entry: The manifest entry.
        folder: The graph folder.

    Returns:
        ``(problems, hashes)``: the violated checks (empty when the graph may load) and the hashes it computed: the
        file SHA-256 of the three splits as ``<split>.txt`` and the content hashes as ``<split>_content``.
    """
    folder = Path(folder)
    problems, hashes = [], {}
    for split in SPLITS:
        path = folder / f'{split}.txt'
        if not path.is_file():
            problems.append(f'{entry.key}: missing {split}.txt')
            continue
        hashes[f'{split}.txt'] = file_sha256(path)
        want = entry.sha256.get(split)
        if want is not None and hashes[f'{split}.txt'] != want:
            problems.append(f'{entry.key}: the SHA-256 of {split}.txt differs from the manifest')
    for split, want in sorted(entry.content_sha256.items()):
        path = folder / f'{split}.txt'
        if path.is_file():
            hashes[f'{split}_content'] = content_sha256(path)
            if hashes[f'{split}_content'] != want:
                problems.append(f'{entry.key}: the content of {split}.txt differs from the manifest')
    if not any(entry.sha256.get(s) for s in SPLITS) and not entry.content_sha256:
        problems.append(f'{entry.key}: the manifest records no hash of this graph')
    return problems, hashes


# --------------------------------------------------------------------------------------------------------- models

@dataclass(frozen=True)
class FamilyPins:
    """The research exports of one ProofPFN family.

    Attributes:
        arm: The arm (``S``, ``R``, ``M``, ``N``, ``HT``, ``CU``, ``UL``; ``S0``, ``DT``, ``PV`` for rows-only arms).
        key: The registered system key (the row key prefix).
        research_run: The run folder of the research exports.
        export: The update of the evaluated export.
        weights: The weights inside the export (``decay/ema``).
        export_sha256: Seed to the SHA-256 of the export file, or ``None`` while the pins are pending.
        loadable: False for the rows-only arms, whose exports the clean code cannot load.
    """

    arm: str
    key: str
    research_run: str
    export: int
    weights: str
    export_sha256: Optional[Mapping[int, str]]
    loadable: bool


@dataclass(frozen=True)
class ReferencePin:
    """A reference checkpoint.

    Attributes:
        key: The system key.
        path: The file, relative to the checkpoint root, or ``None`` for an external adapter that finds its own files.
        sha256: The file SHA-256.
    """

    key: str
    path: Optional[str]
    sha256: str


@dataclass(frozen=True)
class ModelPins:
    """The pins of ``models.json``.

    Attributes:
        families: Arm to family pins (ported and rows-only arms).
        references: System key to reference pin.
        sha256: The SHA-256 of the file.
    """

    families: Mapping[str, FamilyPins]
    references: Mapping[str, ReferencePin]
    sha256: str

    def family_of_key(self, key: str) -> Optional[FamilyPins]:
        """The family registered under a system key, or ``None``."""
        return next((f for f in self.families.values() if f.key == key), None)

    def fv1_models(self) -> dict[str, dict[str, dict[str, str]]]:
        """The ``--models`` input of the registered ``fv1_tables.py``: ``{"models": {hash key: {seed or "model": sha256}}}``.

        Families with pending pins are left out (as in the research file); references enter by their file hash.
        """
        models: dict[str, dict[str, str]] = {}
        for family in self.families.values():
            if family.export_sha256 is not None:
                models[family.key] = {str(s): family.export_sha256[s] for s in SEEDS}
        for ref in self.references.values():
            models[ref.key] = {'model': ref.sha256}
        return {'models': dict(sorted(models.items()))}


@lru_cache(maxsize=None)
def _read_models(path: str) -> ModelPins:
    raw = Path(path).read_bytes()
    doc = json.loads(raw)
    if doc.get('format') != MODELS_FORMAT:
        raise ManifestError(f'{path} is not a {MODELS_FORMAT} file')
    families = {}
    for loadable, section in ((True, 'families'), (False, 'rows_only')):
        for arm, item in doc[section].items():
            pins = item['export_sha256']
            if pins is not None and set(pins) != {str(s) for s in SEEDS}:
                raise ManifestError(f'{path}: {arm} needs pins for the seeds {SEEDS}')
            families[arm] = FamilyPins(arm=arm, key=item['key'], research_run=item['research_run'], export=int(item['export']),
                                       weights=item['weights'], loadable=loadable,
                                       export_sha256=None if pins is None else MappingProxyType({int(s): v for s, v in pins.items()}))
    keys = [f.key for f in families.values()] + list(doc['references'])
    if len(set(keys)) != len(keys):
        raise ManifestError(f'{path}: system keys must be unique')
    references = {key: ReferencePin(key=key, path=item.get('path'), sha256=item['sha256']) for key, item in doc['references'].items()}
    return ModelPins(families=MappingProxyType(families), references=MappingProxyType(references), sha256=hashlib.sha256(raw).hexdigest())


def read_models(path: Optional[PathLike] = None) -> ModelPins:
    """The model pins (``manifests/models.json`` by default)."""
    return _read_models(str(Path(path) if path is not None else MANIFESTS / 'models.json'))


# --------------------------------------------------------------------------------------------------------- builders

def _manifest(name: str, about: str, sample_rule: str, sources: Mapping[str, str], graphs: list[dict]) -> dict:
    return dict(format=FORMAT, name=name, about=about, blind=False, sample_rule=sample_rule, sources=dict(sources), graphs=graphs)


def _audit_entry(g: Mapping[str, Any], inventory: Mapping[str, Mapping[str, Any]], view: Optional[str] = None) -> dict:
    code = GROUP_CODE.get(g['group']) if g['group'] is not None else None
    if view is None:
        if code is None:
            raise ManifestError(f"{g['key']}: the audit gives no group, so the graph has no view")
        view = VIEW_OF_CODE[code]
    inv = inventory[g['key']]
    return dict(key=g['key'], name=g['name'], path=g['path'], collection=g['collection'], view=view,
                code=code, group=g['group'], verdict=g['verdict'], weight=g['weight'], origin_class=g['origin_class'],
                suite_id=g['suite_id'], size=dict(entities=g['size']['entities'], inference_edges=g['size']['inference_edges']),
                sha256=dict.fromkeys(SPLITS), content_sha256={s: inv[f'sha_{s}'] for s in CONTENT_SPLITS}, counts=None, tail_only_view=None)


def build_dev121(audit: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]], suite: Mapping[str, Any],
                 sources: Mapping[str, str]) -> dict:
    """The ``dev121`` manifest from ``graph-audit-v1/graphs.json``, its ``stats/inventory.json`` and ``p0-suite-v1/manifest.json``.

    The 121 graphs with an audit group, in audit order. The content hashes come from the inventory; the file hashes
    from the suite manifest where it records the same folder.

    Raises:
        ManifestError: When the audit does not hold 121 grouped graphs with unique names (as ``ag_eval`` asserts).
    """
    inv = {e['key']: e for e in inventory}
    files = {g['folder']: g['files'] for g in suite['graphs'].values()}
    graphs = []
    for g in audit['graphs']:
        if g['group'] is None:
            continue
        if 'lockbox3' in g['path']:
            raise ManifestError(f"{g['key']}: an audited path names lockbox3")
        entry = _audit_entry(g, inv)
        if g['path'] in files:
            entry['sha256'] = {s: files[g['path']][s]['sha256'] for s in SPLITS}
        graphs.append(entry)
    if len(graphs) != 121 or len({g['name'] for g in graphs}) != 121:
        raise ManifestError(f'the audit holds {len(graphs)} grouped graphs, not 121 with unique names')
    about = ('The 121 audited development graphs of graph-audit-v1 (16 October 2026): views main (91 graphs of the groups E, L, '
             'B, S, O, R; weights sum to 1), contamination (29) and apart (UMLS). Content hashes of train.txt and test.txt '
             'from the audit inventory (checked before a graph loads); file hashes from the standard suite manifest where it '
             'records the same folder, else null (the research harness checked the content hashes only).')
    return _manifest('dev121', about, 'first-1000', sources, graphs)


def build_t1(suite: Mapping[str, Any], audit: Mapping[str, Any], sources: Mapping[str, str]) -> dict:
    """The ``t1`` manifest: the 54 zero-shot graphs of ``p0-suite-v1/manifest.json`` (its three pretraining graphs left out).

    Names are the suite ids with ``:`` replaced by ``_``. ``size`` (the inference-graph statistics of Flock's rule) comes
    from the audit entry of the same folder.

    Raises:
        ManifestError: When the suite does not hold 54 zero-shot graphs, or a folder has no audit entry.
    """
    by_path = {g['path']: g for g in audit['graphs']}
    graphs = []
    for sid, g in suite['graphs'].items():
        if g['group'] == 'pretrain':
            continue
        audited = by_path.get(g['folder'])
        if audited is None or audited['suite_id'] != sid:
            raise ManifestError(f'{sid}: no audit entry with this suite id for {g["folder"]}')
        name = sid.replace(':', '_')
        graphs.append(dict(key=f't1/{name}', name=name, path=g['folder'], collection='t1', view=g['group'], code=None, group=g['group'],
                           verdict=None, weight=0.0, origin_class=None, suite_id=sid,
                           size=dict(entities=audited['size']['entities'], inference_edges=audited['size']['inference_edges']),
                           sha256={s: g['files'][s]['sha256'] for s in SPLITS}, content_sha256={}, counts=dict(g['counts']),
                           tail_only_view=g['tail_only_view']))
    if len(graphs) != 54:
        raise ManifestError(f'the suite holds {len(graphs)} zero-shot graphs, not 54')
    about = ('T1, the standard zero-shot suite p0-suite-v1 (built under the official ULTRA conventions): 54 graphs in the groups '
             'trans, ind_e and ind_er, every test row ranked in both directions. File SHA-256 and counts from the suite manifest; '
             'the suite id is the key of the official Flock walk table; size from the audit entry of the same folder.')
    return _manifest('t1', about, 'all', sources, graphs)


def build_gate(audit: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]], recorded: Mapping[str, Mapping[str, str]],
               sources: Mapping[str, str]) -> dict:
    """The ``gate`` manifest: the graphs of ``GATE_GRAPHS`` with the audit fields, content hashes and recorded file hashes.

    Args:
        audit: ``graph-audit-v1/graphs.json``.
        inventory: ``graph-audit-v1/stats/inventory.json``.
        recorded: Graph key to the file SHA-256 per split (the files that the golden runs hashed).
        sources: The inputs and their SHA-256.

    Raises:
        ManifestError: When a gate graph lacks recorded file hashes.
    """
    inv = {e['key']: e for e in inventory}
    by_key = {g['key']: g for g in audit['graphs']}
    graphs = []
    for key in GATE_GRAPHS:
        if set(recorded.get(key, {})) != set(SPLITS):
            raise ManifestError(f'{key}: no recorded file hashes')
        entry = _audit_entry(by_key[key], inv, view='gate')
        entry['sha256'] = {s: recorded[key][s] for s in SPLITS}
        graphs.append(entry)
    about = ('The gate graphs of the exact-rank goldens G6 and G7: UMLS, NELLInductive_v1 and grail-WN18RR_v1. The audit excludes '
             'grail-WN18RR_v1 (X2: a copy of ultra-suite-v1/WN18RRInductive_v1), so it is listed here and not in dev121. File '
             'SHA-256 as the golden runs hashed the files; content hashes from the audit inventory.')
    return _manifest('gate', about, 'first-1000', sources, graphs)


def dumps(doc: Mapping[str, Any]) -> str:
    """The canonical text of a manifest file."""
    return json.dumps(doc, indent=1) + '\n'
