"""Frozen graph sets of the ProofPFN benchmark, resolved without PyTorch.

A graph manifest (``manifests/<name>.json``) lists graphs with their folders and the hashes that each graph must match
before it loads:

- ``ultra54``: the 54 zero-shot graphs of the standard suite, built under the official ULTRA conventions, with their
  suite ids (the keys of Flock's walk table), file hashes and counts. Views: the suite groups ``trans``, ``ind_e`` and
  ``ind_er``. Every test row is ranked.
- ``heldout44``: the 44 graphs of the held-out collection of the thesis, with file hashes and counts. Views:
  ``published`` (16 graphs) and ``converted`` (28 graphs from public sources). Each test file holds the 1,000 sampled
  test facts of the collection, and every row is ranked.

A manifest file can also be given by its path, for example for a graph set of your own. ``sample_rule`` is ``all``
(every test row) or ``first-1000`` (``data.ranked_positions``).

``tests/test_manifests.py`` derives the manifests from the inputs named in their ``sources`` and compares them with
the committed files when the inputs are available.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Union

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = Path(__file__).resolve().parent / 'manifests'
FORMAT = 'proofpfn-graph-manifest-v2'
SPLITS = ('train', 'valid', 'test')
CONTENT_SPLITS = ('train', 'test')
COUNTS = ('entities', 'relations', 'inference_rows', 'valid_rows', 'test_rows')
SAMPLE, SAMPLE_SEED = 1000, 0
SAMPLE_RULES = {
    'first-1000': ('test rows <= 1000: all rows; else the first occurrence of every distinct test fact in file order, '
                   'sorted(random.Random(0).sample(pool, 1000)); index = position in test.txt'),
    'all': 'every test row; index = position in test.txt',
}
PathLike = Union[str, Path]


class InputError(ValueError):
    """Wrong input: the command line exits with code 2 and one ``error:`` line instead of a traceback."""


class ManifestError(InputError):
    """A manifest that is missing or malformed, or a graph that fails its hash check."""


@dataclass(frozen=True)
class GraphEntry:
    """One graph of a manifest.

    Attributes:
        name: The graph name, the stem of its result files.
        path: The folder with train.txt, valid.txt and test.txt, relative to the data root.
        view: The view that holds the graph (a suite group, for example).
        suite_id: The standard-suite id (the key of Flock's walk table), or ``None``.
        size: ``entities`` and ``inference_edges`` of ``train.txt`` (distinct entities, distinct triples).
        sha256: The file SHA-256 per split, ``None`` where the manifest records none.
        content_sha256: The content hash per split (``train``, ``test``), where recorded (``content_sha256``).
        counts: The counts of ``COUNTS`` over the graph's files, or ``None``.
    """

    name: str
    path: str
    view: str
    suite_id: Optional[str]
    size: Mapping[str, int]
    sha256: Mapping[str, Optional[str]]
    content_sha256: Mapping[str, str]
    counts: Optional[Mapping[str, int]]

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> 'GraphEntry':
        """An entry from its manifest JSON object.

        Args:
            value: An object of the manifest's ``graphs`` list. It needs ``name``, ``path``, ``view``, ``size`` and
                ``sha256``. A split missing from ``sha256`` becomes ``None``.

        Returns:
            The entry, with read-only mappings.

        Raises:
            ManifestError: When a field is missing or malformed.
        """
        try:
            entry = cls(name=value['name'], path=value['path'], view=value['view'], suite_id=value.get('suite_id'),
                        size=MappingProxyType(dict(value['size'])), sha256=MappingProxyType({s: value['sha256'].get(s) for s in SPLITS}),
                        content_sha256=MappingProxyType(dict(value.get('content_sha256') or {})),
                        counts=MappingProxyType(dict(value['counts'])) if value.get('counts') is not None else None)
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            raise ManifestError(f'malformed graph entry {value.get("name") if isinstance(value, Mapping) else value!r}: {error!r}') from None
        if Path(entry.path).is_absolute() or '..' in Path(entry.path).parts or not entry.name or '/' in entry.name:
            raise ManifestError(f'{entry.name}: the path must be relative to the data root and the name a plain file stem')
        if set(entry.content_sha256) - set(CONTENT_SPLITS):
            raise ManifestError(f'{entry.name}: content hashes exist for train and test only')
        return entry


@dataclass(frozen=True)
class Manifest:
    """A frozen graph set.

    Attributes:
        name: The manifest name (``ultra54``, ``heldout44`` or the name of your own manifest); result folders carry it.
        sample_rule: ``first-1000`` or ``all`` (``SAMPLE_RULES``).
        entries: The graphs, in manifest order.
        sha256: The SHA-256 of the manifest file.
    """

    name: str
    sample_rule: str
    entries: tuple[GraphEntry, ...]
    sha256: str

    def __post_init__(self) -> None:
        names = [e.name for e in self.entries]
        if len(set(names)) != len(names):
            raise ManifestError(f'{self.name}: graph names must be unique')
        if self.sample_rule not in SAMPLE_RULES:
            raise ManifestError(f'{self.name}: unknown sample rule {self.sample_rule!r}')
        if not isinstance(self.name, str) or not self.name or not all(c.isalnum() or c in '-_' for c in self.name):
            raise ManifestError(f'manifest name {self.name!r}: letters, digits, - and _')

    @property
    def views(self) -> dict[str, tuple[str, ...]]:
        """View name to graph names, with ``all``."""
        out: dict[str, list[str]] = {}
        for entry in self.entries:
            out.setdefault(entry.view, []).append(entry.name)
        return {'all': tuple(e.name for e in self.entries), **{k: tuple(v) for k, v in out.items()}}

    def entry(self, graph: str) -> GraphEntry:
        """The entry of a graph name.

        Args:
            graph: A graph name.

        Returns:
            The entry.

        Raises:
            KeyError: For a graph outside the manifest.
        """
        for entry in self.entries:
            if graph == entry.name:
                return entry
        raise KeyError(f'{graph!r} is not a graph of manifest {self.name}')

    def select(self, view: Optional[str] = None, graphs: Optional[Iterable[str]] = None) -> list[GraphEntry]:
        """The entries of explicit graphs (in the given order) or of a view; no duplicates.

        Args:
            view: A view name (``views``). It selects the graphs when ``graphs`` is ``None`` or empty, and filters
                them otherwise.
            graphs: Graph names, in output order.

        Returns:
            The entries, each once, in the order of their first appearance in ``graphs`` or in the view.

        Raises:
            KeyError: For an unknown graph or view.
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
        return list({e.name: e for e in wanted}.values())


def read_manifest(source: PathLike) -> Manifest:
    """Open a graph manifest by name (``manifests/<name>.json``) or path.

    Args:
        source: A manifest name or a JSON file.

    Returns:
        The manifest.

    Raises:
        ManifestError: For a missing or malformed manifest.
    """
    path = Path(source)
    if not path.suffix and not path.exists():
        path = MANIFESTS / f'{source}.json'
    if not path.is_file():
        raise ManifestError(f'{source}: neither a manifest of {MANIFESTS.relative_to(REPO)} nor a manifest file')
    try:
        raw = path.read_bytes()
        doc = json.loads(raw)
    except (OSError, ValueError) as error:   # unreadable, not UTF-8 or not JSON
        raise ManifestError(f'cannot read {path}: {error}') from None
    if not isinstance(doc, dict) or doc.get('format') != FORMAT:
        raise ManifestError(f'{path} is not a {FORMAT} file')
    try:
        return Manifest(name=doc['name'], sample_rule=doc['sample_rule'], entries=tuple(GraphEntry.from_json(g) for g in doc['graphs']),
                        sha256=hashlib.sha256(raw).hexdigest())
    except (KeyError, TypeError) as error:
        raise ManifestError(f'{path}: a malformed manifest ({error!r})') from None


# --------------------------------------------------------------------------------------------------------- hashes

def file_sha256(path: PathLike) -> str:
    """SHA-256 of a file's bytes.

    Args:
        path: The file.

    Returns:
        The hex digest.
    """
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def content_sha256(path: PathLike) -> str:
    """The content hash of a triple file: SHA-256 of its sorted distinct triples, tab-joined, one per line.

    A line splits at tabs if it has one, else at whitespace; lines without exactly three fields are skipped.

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
            problems.append(f'{entry.name}: missing {split}.txt')
            continue
        hashes[f'{split}.txt'] = file_sha256(path)
        want = entry.sha256.get(split)
        if want is not None and hashes[f'{split}.txt'] != want:
            problems.append(f'{entry.name}: the SHA-256 of {split}.txt differs from the manifest')
    for split, want in sorted(entry.content_sha256.items()):
        path = folder / f'{split}.txt'
        if path.is_file():
            hashes[f'{split}_content'] = content_sha256(path)
            if hashes[f'{split}_content'] != want:
                problems.append(f'{entry.name}: the content of {split}.txt differs from the manifest')
    if not any(entry.sha256.get(s) for s in SPLITS) and not entry.content_sha256:
        problems.append(f'{entry.name}: the manifest records no hash of this graph')
    return problems, hashes
