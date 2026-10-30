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
it. Any other manifest is refused when it names a blind folder: a path with a part ``fresh-v1`` or ``lockbox3``, or a
folder that resolves (links followed) below ``BLIND_FOLDERS`` of the repository. This branch holds no blind manifest.

``models.json`` pins the research exports of the ProofPFN families (seeds 42, 43 and 44) and the reference checkpoints.
HT's pins come from the research ``models-pcuht.json``, written after HT's training. A family whose pins are ``null``
is refused under its registered key.

``tests/test_manifests.py`` derives the graph manifests from the research inputs named in each manifest's ``sources``
and compares them with the committed files when the inputs are available.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Union

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = Path(__file__).resolve().parent / 'manifests'
FORMAT = 'proofpfn-graph-manifest-v1'
MODELS_FORMAT = 'proofpfn-models-v1'
SPLITS = ('train', 'valid', 'test')
CONTENT_SPLITS = ('train', 'test')
SEEDS = (42, 43, 44)
BLIND_FOLDERS = ('KGs/fresh-v1', 'KGs/lockbox3')
"""The folders of the blind test sets, relative to the repository root."""
BLIND_NAMES = tuple(Path(folder).name for folder in BLIND_FOLDERS)
SAMPLE, SAMPLE_SEED = 1000, 0
SAMPLE_RULES = {
    'first-1000': ('test rows <= 1000: all rows (fair_eval); else the first occurrence of every distinct test fact in file '
                   'order, sorted(random.Random(0).sample(pool, 1000)); index = position in test.txt'),
    'all': 'every test row; index = position in test.txt',
}
PathLike = Union[str, Path]


class InputError(ValueError):
    """Wrong input: the command line exits with code 2 and one ``error:`` line instead of a traceback."""


class ManifestError(InputError):
    """A manifest that is missing or malformed, or a graph that fails its hash check."""


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

        Args:
            value: An object of the manifest's ``graphs`` list. It needs ``key``, ``name``, ``path``, ``collection``,
                ``view``, ``size`` and ``sha256``. A split missing from ``sha256`` becomes ``None``, and ``weight``
                defaults to 0.0.

        Returns:
            The entry, with read-only mappings.

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
        """The entry's fields for the ``allgraph`` record of a row file
        (with the suite's tail-only flag where it has one).

        Returns:
            ``key``, ``collection``, ``path``, ``group``, ``verdict``, ``weight``, ``origin_class`` and
            ``suite_id``, then ``tail_only_view`` when it is not ``None``.
        """
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
        blind: True for a confirmatory test set, which opens only with its release flag.
        sample_rule: ``first-1000`` or ``all`` (``SAMPLE_RULES``).
        entries: The graphs, in manifest order.
        sha256: The SHA-256 of the manifest file.
    """

    name: str
    blind: bool
    sample_rule: str
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

        Args:
            graph: A graph name or key.

        Returns:
            The first entry in manifest order whose name or key is ``graph``.

        Raises:
            KeyError: For a graph outside the manifest.
        """
        for entry in self.entries:
            if graph in (entry.name, entry.key):
                return entry
        raise KeyError(f'{graph!r} is not a graph of manifest {self.name}')

    def select(self, view: Optional[str] = None, graphs: Optional[Iterable[str]] = None, codes: Optional[Iterable[str]] = None) -> list[GraphEntry]:
        """The entries of explicit graphs (names or keys, in the given order) or of a view, then of the audit group
        codes; no duplicates.

        Args:
            view: A view name (``views``). It selects the graphs when ``graphs`` is ``None`` or empty, and filters
                them otherwise.
            graphs: Graph names or keys, in output order.
            codes: The audit group codes that filter the entries. ``None`` applies no filter.

        Returns:
            The entries, each once, in the order of their first appearance in ``graphs`` or in the view.

        Raises:
            KeyError: For an unknown graph, view or audit group code.
            ValueError: When neither graphs nor a view is given.
        """
        if view and view not in self.views:
            raise KeyError(f'{view!r} is not a view of {self.name}: {sorted(self.views)}')
        if graphs:
            wanted = [self.entry(g) for g in graphs]
            if view:
                wanted = [e for e in wanted if e.name in self.views[view]]
        elif view:
            wanted = [self.entry(g) for g in self.views[view]]
        else:
            raise ValueError('choose graphs or a view')
        if codes is not None:
            known, allowed = {e.code for e in self.entries if e.code}, set(codes)
            if allowed - known:
                raise KeyError(f'{sorted(allowed - known)} are no audit group codes of {self.name}: {sorted(known)}')
            wanted = [e for e in wanted if e.code in allowed]
        return list({e.key: e for e in wanted}.values())

    def folder(self, entry: GraphEntry, root: PathLike) -> Path:
        """The folder of an entry below the data root (a blind folder only for a blind manifest).

        A development manifest is refused when the folder or its resolved path (links followed) names a blind folder
        (``in_blind_folder``), or when the resolved path lies below a resolved ``BLIND_FOLDERS`` folder.

        Args:
            entry: The graph.
            root: The data root that the manifest paths are relative to.

        Returns:
            The folder ``<root>/<entry.path>``, with links not resolved.

        Raises:
            BlindSetError: For a blind folder named by a development manifest.
        """
        folder = Path(root) / entry.path
        if not self.blind:
            resolved = folder.resolve()
            if in_blind_folder(folder) or in_blind_folder(resolved) or any(resolved.is_relative_to((REPO / b).resolve()) for b in BLIND_FOLDERS):
                raise BlindSetError(f'{entry.key}: {folder} is a blind folder')
        return folder


def in_blind_folder(path: PathLike) -> bool:
    """Whether a path names a blind folder: one of its parts is the name of a ``BLIND_FOLDERS`` folder (``fresh-v1``,
    ``lockbox3``).

    Args:
        path: The path, checked as given (links not followed).

    Returns:
        True when one of its parts is a name of ``BLIND_NAMES``.
    """
    return any(part in BLIND_NAMES for part in Path(path).parts)


def read_manifest(source: PathLike, release: Optional[str] = None) -> Manifest:
    """Open a graph manifest by name (``manifests/<name>.json``) or path.

    Args:
        source: A manifest name or a JSON file.
        release: The name of the blind manifest that this run may open; ``None`` refuses every blind manifest.

    Returns:
        The manifest.

    Raises:
        BlindSetError: For a blind manifest whose name is not ``release``, or a development manifest that names a blind
            folder.
        ManifestError: For a missing or malformed manifest.
    """
    path = Path(source)
    if not path.suffix and not path.exists():
        path = MANIFESTS / f'{source}.json'
    if not path.is_file():
        raise ManifestError(f'{source}: neither a manifest of {MANIFESTS.relative_to(REPO)} nor a manifest file')
    doc, digest = _read_json(path, FORMAT)
    blind = bool(doc.get('blind', False))
    if blind and release != doc.get('name'):
        raise BlindSetError(f"{doc.get('name')} is a blind test set; it opens only with the release flag naming it")
    try:
        return Manifest(name=doc['name'], blind=blind, sample_rule=doc['sample_rule'], entries=tuple(GraphEntry.from_json(g) for g in doc['graphs']),
                        sha256=digest)
    except (KeyError, TypeError) as error:
        raise ManifestError(f'{path}: a malformed manifest ({error!r})') from None


def _read_json(path: Path, kind: str) -> tuple[dict[str, Any], str]:
    """The JSON object of a manifest file of a format, and the file's SHA-256.

    Raises:
        ManifestError: When the file cannot be read, is not JSON or is of another format.
    """
    try:
        raw = path.read_bytes()
        doc = json.loads(raw)
    except (OSError, ValueError) as error:   # missing, unreadable, not UTF-8 or not JSON
        raise ManifestError(f'cannot read {path}: {error}') from None
    if not isinstance(doc, dict) or doc.get('format') != kind:
        raise ManifestError(f'{path} is not a {kind} file')
    return doc, hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------------------------------------- hashes

def file_sha256(path: PathLike) -> str:
    """SHA-256 of a file's bytes (``dicee.synthetic_kg.digests.file_sha256``, here without importing PyTorch).

    Args:
        path: The file.

    Returns:
        The hex digest.
    """
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def content_sha256(path: PathLike) -> str:
    """The audit's content hash of a triple file: SHA-256 of its sorted distinct triples, tab-joined, one per line.

    A line splits at tabs if it has one, else at whitespace; lines without exactly three fields are skipped (the rule of
    ``graph-audit-v1``, restated by ``allgraph-v1/ag_eval.content_sha``).

    Args:
        path: The triple file, read as UTF-8 with undecodable bytes replaced.

    Returns:
        The hex digest.
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
        export: The update of the evaluated export.
        weights: The weights inside the export (``decay/ema``).
        export_sha256: Seed to the SHA-256 of the export file, or ``None`` while the pins are pending.
        loadable: False for the rows-only arms, whose exports the clean code cannot load.
    """

    arm: str
    key: str
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
        external: Adapter name to the pins of its external files (``kgpfn``, ``merit``, ``anyburl``).
    """

    families: Mapping[str, FamilyPins]
    references: Mapping[str, ReferencePin]
    external: Mapping[str, Mapping[str, Any]]

    def family_of_key(self, key: str) -> Optional[FamilyPins]:
        """The family registered under a system key, or ``None``.

        Args:
            key: A system key.

        Returns:
            The family pins whose ``key`` is ``key``, or ``None`` when no family has it.
        """
        return next((f for f in self.families.values() if f.key == key), None)

    def fv1_models(self) -> dict[str, dict[str, dict[str, str]]]:
        """The ``--models`` input of the registered ``fv1_tables.py``:
        ``{"models": {hash key: {seed or "model": sha256}}}``.

        Families with pending pins are left out (as in the research file); references enter by their file hash.

        Returns:
            The object, with its hash keys sorted. A family enters under its system key and maps each seed of
            ``SEEDS``, as a string, to the SHA-256 of its export. A reference enters under its key and maps ``"model"``
            to the SHA-256 of its file.
        """
        models: dict[str, dict[str, str]] = {}
        for family in self.families.values():
            if family.export_sha256 is not None:
                models[family.key] = {str(s): family.export_sha256[s] for s in SEEDS}
        for ref in self.references.values():
            models[ref.key] = {'model': ref.sha256}
        return {'models': dict(sorted(models.items()))}


def read_models(path: Optional[PathLike] = None) -> ModelPins:
    """The model pins (``manifests/models.json`` by default).

    Args:
        path: The models file, or ``None`` for ``manifests/models.json``.

    Returns:
        The pins of the sections ``families`` and ``rows_only`` (by arm), ``references`` (by system key) and
        ``external`` (by adapter), as read-only mappings.

    Raises:
        ManifestError: For a missing or malformed file.
    """
    path = Path(path) if path is not None else MANIFESTS / 'models.json'
    doc, _ = _read_json(path, MODELS_FORMAT)
    families = {}
    try:
        for loadable, section in ((True, 'families'), (False, 'rows_only')):
            for arm, item in doc[section].items():
                pins = item['export_sha256']
                if pins is not None and set(pins) != {str(s) for s in SEEDS}:
                    raise ManifestError(f'{path}: {arm} needs pins for the seeds {SEEDS}')
                families[arm] = FamilyPins(arm=arm, key=item['key'], export=int(item['export']), weights=item['weights'], loadable=loadable,
                                           export_sha256=None if pins is None else MappingProxyType({int(s): v for s, v in pins.items()}))
        references = {key: ReferencePin(key=key, path=item.get('path'), sha256=item['sha256']) for key, item in doc['references'].items()}
        external = {name: MappingProxyType(dict(item)) for name, item in doc.get('external', {}).items()}
    except ManifestError:
        raise
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ManifestError(f'{path}: malformed model pins ({error!r})') from None
    keys = [f.key for f in families.values()] + list(references)
    if len(set(keys)) != len(keys):
        raise ManifestError(f'{path}: system keys must be unique')
    return ModelPins(families=MappingProxyType(families), references=MappingProxyType(references), external=MappingProxyType(external))
