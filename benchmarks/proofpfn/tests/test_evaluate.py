"""The evaluation runner on the CPU: ranks, the result file, stored results, the summary, and official Flock."""
import shutil
import statistics

import numpy as np
import pytest
import torch

from benchmarks.proofpfn import evaluate as ev
from benchmarks.proofpfn import manifests as mf
from benchmarks.proofpfn import systems as sy
from dicee.evaluation._filtering import FilteredRanker
from tests.proofpfn_fixtures import clean_checkpoints

from .helpers import tiny_graph, write_graph, write_manifest


def research_rank(v: torch.Tensor, target: int, mask: torch.Tensor) -> int:
    """``lockbox-v5-runtime/lb5_eval.rank``: the pessimistic rank of the thesis runs."""
    return int(((v >= v[target]) & mask).sum())


def test_pessimistic_ranker_equals_the_research_rank():
    rng = np.random.default_rng(0)
    ranker = FilteredRanker(tie_policy='pessimistic')
    for _ in range(400):
        n = int(rng.integers(2, 40))
        v = torch.tensor(np.round(rng.normal(size=n), 1))        # rounding makes ties
        anchor, target = (int(x) for x in rng.integers(0, n, 2))
        if rng.random() < 0.5:
            v[anchor] = -float('inf')                            # also when the target is the anchor
        others = sorted(set(rng.integers(0, n, int(rng.integers(0, n))).tolist()) - {target})
        mask = torch.ones(n, dtype=torch.bool)
        mask[torch.tensor(others, dtype=torch.long)] = False
        assert ranker.rank(v, target, others, exclude_target=False) == research_rank(v, target, mask)


def test_metrics():
    metrics = ev.metrics_of([False, True, False, True], [1, 2, 4, 20])
    assert metrics['both'] == {'MRR': (1 + 0.5 + 0.25 + 0.05) / 4, 'H@1': 0.25, 'H@3': 0.5, 'H@10': 0.75}
    assert metrics['tail']['MRR'] == (1 + 0.25) / 2 and metrics['head']['H@10'] == 0.5
    assert set(ev.metrics_of([False], [3])) == {'both', 'tail'}


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    """A set of two tiny graphs in two views, the checkpoints of two variants, the systems and one run of all of them.

    TRIX and ULTRA-3g join when their checkpoints are available. Official Flock takes 1.5 s per query group on the CPU,
    so it has its own test on a smaller graph.
    """
    root = tmp_path_factory.mktemp('tiny')
    folders = {'g': tiny_graph(root / 'KGs' / 'g'), 'h': tiny_graph(root / 'KGs' / 'h', seed=1)}
    manifest = mf.read_manifest(write_manifest(root / 'tiny.json', root, folders, views={'g': 'a', 'h': 'b'}))
    systems: list = sy.checkpoints(f"ts={clean_checkpoints(root, 'PCUltra-S')}") + sy.checkpoints(f"tu={clean_checkpoints(root, 'ULTRA-S').format(seed=42)}")
    for label in ('trix', 'ultra3g'):
        reference = sy.reference(label)
        if reference.path.is_file():
            systems.append(reference)
    for entry in manifest.entries:
        assert set(ev.evaluate_graph(manifest, entry, systems, root / 'out', root=root, device='cpu').values()) == {'done'}
    return root, manifest, systems


def test_result_file(tiny):
    root, manifest, systems = tiny
    entry = manifest.entry('g')
    assert set(ev.evaluate_graph(manifest, entry, systems, root / 'out', root=root, device='cpu').values()) == {'exists'}
    facts = len((root / 'KGs' / 'g' / 'test.txt').read_text().splitlines())
    for system in systems:
        result = ev.read_result(ev.result_path(root / 'out', manifest, system.label, entry))
        assert (result['label'], result['graph'], result['manifest']) == (system.label, 'g', dict(name='tiny', sha256=manifest.sha256))
        assert len(result['rank']) == len(result['index']) == len(result['head']) == 2 * facts == 2 * result['ranked_facts']
        assert result['metrics'] == ev.metrics_of(result['head'], result['rank']) and result['model_sha256'] == system.sha256()
        assert result['files']['train.txt'] == mf.file_sha256(root / 'KGs' / 'g' / 'train.txt') and result['system'] == system.spec()
        assert 'flock_setting' not in result and result['provenance']['deterministic'] and result['provenance']['threads'] == 2
    assert ev.result_path(root / 'out', manifest, 'ts.seed42', entry) == root / 'out' / 'tiny' / 'ts.seed42' / 'g.json'
    with pytest.raises(mf.InputError, match='is not a proofpfn-ranks-v1 file'):
        ev.read_result(root / 'tiny.json')


def test_a_new_system_runs_beside_stored_results(tiny, tmp_path):
    root, manifest, systems = tiny
    shutil.copytree(root / 'out', tmp_path / 'out')
    entry = manifest.entry('g')
    other = sy.checkpoints(f"tc={root / 'PCUltra-S-43.pt'}")   # the same file under another label
    assert ev.evaluate_graph(manifest, entry, [systems[0], *other], tmp_path / 'out', root=root, device='cpu') == {'ts.seed42': 'exists', 'tc': 'done'}
    a, b = (ev.read_result(ev.result_path(tmp_path / 'out', manifest, label, entry)) for label in ('tc', 'ts.seed43'))
    assert a['rank'] == b['rank'] and a['model_sha256'] == b['model_sha256']
    with pytest.raises(mf.InputError, match='unique'):
        ev.evaluate_graph(manifest, entry, [systems[0], systems[0]], tmp_path / 'twice', root=root, device='cpu')


def test_summary(tiny, tmp_path):
    root, manifest, _ = tiny
    out = tmp_path / 'out'
    shutil.copytree(root / 'out', out)
    (out / 'tiny' / 'tu' / 'h.json').unlink()   # a label without every graph
    summary = ev.summarize(out, manifest, manifest.entries)
    assert summary['graphs'] == ['g', 'h'] and summary['views'] == {'a': ['g'], 'b': ['h']} and summary['direction'] == 'both'
    per_graph, row = summary['per_graph']['ts.seed42'], summary['labels']['ts.seed42']
    assert row['complete'] and row['found'] == 2 and row['a'] == per_graph['g'] and row['b'] == per_graph['h']
    assert row['all']['MRR'] == pytest.approx((per_graph['g']['MRR'] + per_graph['h']['MRR']) / 2)
    partial = summary['labels']['tu']
    assert not partial['complete'] and partial['found'] == 1 and 'a' in partial and 'b' not in partial and 'all' not in partial
    values = [summary['labels'][f'ts.seed{seed}']['all']['MRR'] for seed in (42, 43, 44)]
    group = summary['groups']['ts']
    assert group['seeds'] == [42, 43, 44] and group['all']['MRR']['mean'] == pytest.approx(statistics.fmean(values))
    assert group['all']['MRR']['sd'] == pytest.approx(statistics.stdev(values)) and 'tu' not in summary['groups']
    tail = ev.summarize(out, manifest, [manifest.entry('g')], 'tail')
    assert tail['graphs'] == ['g'] and tail['labels']['tu']['complete']
    result = ev.read_result(ev.result_path(out, manifest, 'tu', manifest.entry('g')))
    assert tail['per_graph']['tu']['g']['MRR'] == result['metrics']['tail']['MRR']
    with pytest.raises(mf.InputError, match='direction'):
        ev.summarize(out, manifest, manifest.entries, 'up')
    changed = mf.read_manifest(write_manifest(tmp_path / 'tiny.json', root, {'g': root / 'KGs' / 'g'}))   # another file of the name "tiny"
    with pytest.raises(mf.InputError, match='another version of manifest tiny'):
        ev.summarize(out, changed, changed.entries)
    assert ev.summarize(tmp_path / 'none', manifest, manifest.entries)['labels'] == {}


def test_official_flock_gives_the_same_ranks_in_two_runs(tmp_path):
    """Official Flock draws the walks of each query from a generator seeded per call."""
    flock = sy.reference('flock')
    if not flock.path.is_file():
        pytest.skip(f'{flock.path} is not available')
    folder = write_graph(tmp_path / 'KGs' / 'g', ['a\tr0\tb', 'b\tr0\tc', 'c\tr1\ta', 'a\tr1\tc', 'b\tr1\td', 'd\tr0\ta'], [],
                         ['a\tr0\tc', 'd\tr1\tb'])
    manifest = mf.read_manifest(write_manifest(tmp_path / 'm.json', tmp_path, {'g': folder}))
    entry = manifest.entry('g')
    for out in ('one', 'two'):
        assert ev.evaluate_graph(manifest, entry, [flock], tmp_path / out, root=tmp_path, device='cpu') == {'flock': 'done'}
    one, two = (ev.read_result(ev.result_path(tmp_path / out, manifest, 'flock', entry)) for out in ('one', 'two'))
    assert one['rank'] == two['rank'] and (one['flock_setting']['walks'], one['flock_setting']['passes']) == (16, 16)


def test_global_settings_are_restored(tiny, tmp_path):
    root, manifest, systems = tiny
    threads, deterministic, rng = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(), torch.get_rng_state()
    ev.evaluate_graph(manifest, manifest.entry('g'), systems[:1], tmp_path / 'settings', root=root, device='cpu')
    assert (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()) == (threads, deterministic)
    assert torch.equal(torch.get_rng_state(), rng)
