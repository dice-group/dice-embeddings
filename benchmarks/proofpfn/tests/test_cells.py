"""The query cells (golden G7: byte-identical queries.csv of the research fv1_cells.py on the gate graphs)."""
import json

import numpy as np
import pytest

from benchmarks.proofpfn import cells
from benchmarks.proofpfn import data as D
from benchmarks.proofpfn import manifests as mf

from .golden import GOLDEN, write_graph


@pytest.mark.parametrize('name', ['UMLS', 'NELLInductive_v1', 'grail-WN18RR_v1'])
def test_cells_equal_the_research_file(name, tmp_path):
    gate = mf.read_manifest('gate')
    entry = gate.entry(name)
    if not gate.folder(entry, mf.REPO).is_dir():
        pytest.skip(f'{entry.path} is not available')
    out = tmp_path / f'queries-{name}.csv'
    meta = cells.write_queries(gate, [entry], out)
    assert out.read_bytes() == (GOLDEN / 'g7_cells' / f'queries-{name}.csv').read_bytes()
    assert meta['out_sha256'] == mf.file_sha256(out) and meta['inputs'][name] == {f'{s}.txt': entry.sha256[s] for s in mf.SPLITS}
    assert json.loads(out.with_name(out.name + '.meta.json').read_text()) == meta


def test_rank_of_takes_the_mean_of_ties():
    score = np.array([3.0, 1.0, 1.0, 1.0, 0.5])
    assert cells.rank_of(score, 1, {0}) == 2.0          # tied with 2 and 3: the mean of ranks 1 to 3
    assert cells.rank_of(score, 4, set()) == 5.0 and cells.rank_of(score, 0, {1, 2, 3}) == 1.0


def test_a_repeated_test_fact_is_refused(tmp_path):
    folder = write_graph(tmp_path / 'g', ['a\tr\tb', 'b\tr\tc'], [], ['c\tr\ta', 'c\tr\ta'])
    with pytest.raises(mf.ManifestError, match='twice'):
        cells.graph_cells(D.read_graph(folder))
