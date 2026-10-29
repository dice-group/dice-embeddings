"""The registered chain on the expected design of golden G8; ``test_registered.py`` runs it as a subprocess.

The working directory is the case folder. The registered files are imported as libraries (their module-level
``sys.path`` entry is part of the frozen files): ``test_fv1_tables.build_tree`` writes the simulated row files of the
expected design (44 confirmatory graphs from 34 sources, 13 block-3 graphs, 1,000 facts per graph), the files of six
drops are removed, ``fv1_tables.main`` writes the tables, ``test_fv1_tables.write_queries`` the cells, and
``fv1_analysis.main analyse`` runs for each B given. Every path given to the scripts is relative to the case folder, as
in the golden run (proofpfn-golden ``tools/g8_analysis.py``, step 3).

Usage (case folder as the working directory): ``python -B -m benchmarks.proofpfn.tests.g8_chain B [B ...]``
"""
import contextlib
import io
import json
import sys
from pathlib import Path

from benchmarks.proofpfn.registered import test_fv1_tables as tft

DROPS = (('FLOCK', 'conv-c22-0'), ('FLOCK', 'conv-c21-0'), ('KGPFN', 'conv-c20-0'), ('KGPFN', 'conv-c01-1'),
         ('MERIT', 'conv-c19-0'), ('ANYBURL', 'pub-b8-0'))
REASON = 'out of memory on GPUs 0 and 1 (simulated drop, proofpfn-golden G8)'


def expected_design() -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, int]]]:
    """``{graph: (block, source)}`` of the 44 confirmatory graphs and ``{graph: (cluster, facts)}`` of the 13 block-3 graphs."""
    graphs = {}
    for s, n in zip(('A1', 'A2', 'A3', 'A4'), (2, 1, 1, 1)):
        graphs.update({f'pub-{s.lower()}-{j}': ('1a', f'SRC-{s}') for j in range(n)})
    graphs['pub-a1-b'] = ('1b', 'SRC-A1')
    for s, n in zip([f'B{i}' for i in range(1, 9)], (2, 2, 1, 1, 1, 1, 1, 1)):
        graphs.update({f'pub-{s.lower()}-{j}': ('1b', f'SRC-{s}') for j in range(n)})
    for i in range(1, 23):
        graphs.update({f'conv-c{i:02d}-{j}': ('2', f'SRC-C{i:02d}') for j in range(2 if i <= 6 else 1)})
    block3 = {f'lb3-{i:02d}': (('dbpedia', 'wikidata', 'yago', 'freebase')[i % 4], 1000) for i in range(13)}
    return graphs, block3


def main(argv: list[str]) -> int:
    """Run the chain in the working directory; ``argv`` holds the bootstrap replicate counts of ``analyse``."""
    fa, ft = tft.fa, tft.ft
    graphs, block3 = expected_design()
    tft.build_tree(Path('.'), graphs, block3, fa.QUERIES_PER_GRAPH)
    for label, graph in DROPS:
        (Path('rows') / label.lower() / f'{graph}.json').unlink()
    with contextlib.redirect_stdout(io.StringIO()):
        ft.main(['--rows', 'rows', '--manifest', 'manifest.json', '--block3', 'block3.json', '--models', 'models.json', '--out-dir', 'tables'])
    tft.write_queries(Path('queries.csv'), Path('tables') / 'ranks.csv')
    Path('gates.json').write_text(json.dumps(dict(kgpfn_parity='pass', n_validity='pass')))
    dropped: dict[str, dict[str, str]] = {}
    for label, graph in DROPS:
        dropped.setdefault(graph, {})[label] = REASON
    Path('dropped.json').write_text(json.dumps(dropped, indent=1, sort_keys=True))
    for b in argv:
        with contextlib.redirect_stdout(io.StringIO()):
            fa.main(['analyse', '--mrr', 'tables/mrr.csv', '--queries', 'queries.csv', '--ranks', 'tables/ranks.csv', '--gates', 'gates.json',
                     '--dropped', 'dropped.json', '--manifest', 'manifest.json', '--cost', 'tables/cost.csv', '--b', b, '--out', f'result-b{b}.json'])
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
