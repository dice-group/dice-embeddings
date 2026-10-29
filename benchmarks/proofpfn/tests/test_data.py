"""The research loader rules: ids, file order, the sample rule, filters, query groups, the self-loop rule, GT2 inputs."""
import random

import numpy as np
import pytest
import torch

from benchmarks.proofpfn import data as D
from benchmarks.proofpfn import manifests as mf

from .golden import fixture, int_digest, write_graph, write_manifest


def test_ids_are_sorted_names_and_rows_keep_file_order(tmp_path):
    folder = write_graph(tmp_path / 'g', ['b\tr2\ta', 'c\tr1\tb', 'b\tr2\ta'], [], ['a\tr3\td', 'NA\tr1\tnull'])
    graph = D.read_graph(folder)
    assert graph.entities == ('NA', 'a', 'b', 'c', 'd', 'null') and graph.relations == ('r1', 'r2', 'r3')
    assert graph.train.tolist() == [[2, 1, 1], [3, 0, 2], [2, 1, 1]]   # duplicates and file order kept
    assert graph.valid.shape == (0, 3) and graph.test.tolist() == [[1, 2, 4], [0, 0, 5]]
    (folder / 'test.txt').write_text('a r3\n')
    with pytest.raises(mf.ManifestError, match='2 fields'):
        D.read_graph(folder)


def test_dicee_kg_reads_other_ids(tmp_path):
    """dicee's pandas reader turns NA and null into missing values; the research loader keeps both entities."""
    from dicee.knowledge_graph import KG
    folder = write_graph(tmp_path / 'g', ['NA\tr1\tb', 'b\tr2\tnull', 'c\tr1\tNA'], ['c\tr2\tb'], ['NA\tr2\tc'])
    kg = KG(dataset_dir=str(folder), backend='pandas', separator=r'\s+', eval_model='train_val_test', training_technique='KvsAll',
            add_reciprocal=False)
    assert kg.entity_to_idx is not None and len(kg.entity_to_idx) == 3 and D.read_graph(folder).num_entities == 4


def test_the_first_1000_sample_rule():
    assert D.sample_positions([(0, 0, i) for i in range(1000)]) is None
    test = [(i % 1500, 0, 0) for i in range(1600)]   # 1,500 distinct facts; rows 1500 to 1599 repeat earlier facts
    positions = D.sample_positions(test)
    assert positions == sorted(random.Random(0).sample(list(range(1500)), 1000))
    assert D.ranked_positions(test, 'all') is None and D.ranked_positions(test, 'first-1000') == positions


def test_query_groups_known_sets_and_rules():
    test = [(0, 1, 2), (0, 1, 3), (4, 0, 2), (5, 1, 2)]
    groups = D.query_groups(test)
    assert groups == [((False, 0, 4), [2]), ((True, 0, 2), [2]), ((False, 1, 0), [0, 1]), ((True, 1, 2), [0, 3]),
                      ((True, 1, 3), [1]), ((False, 1, 5), [3])]
    known = D.known_sets(test)
    assert known[False][1, 0] == {2, 3} and known[True][1, 2] == {0, 5}
    train = [(i, 0, i + 1) for i in range(99)] + [(7, 0, 7)] + [(1, 1, 1), (1, 1, 2)]
    assert D.reflexive_relations(train) == {0, 1}                          # 1 of 100 and 1 of 2 facts are self-loops
    more = train + [(200, 0, 201)]                                         # 1 of 101: below 1%
    assert D.reflexive_relations(more) == {1} and D.reflexive_relations(more, 'any') == {0, 1}
    gt2 = D.gt2_inputs([(0, 0, 1), (2, 0, 1), (2, 0, 3)], 5)
    unseen, g = gt2[False, 0]                                              # tails of relation 0: 1 twice, 3 once
    assert unseen.tolist() == [True, False, True, False, True] and g == (1 + 0.5) / (3 + 1)
    assert D.inference_stats(torch.tensor([[0, 0, 1], [0, 0, 1], [2, 1, 1]])) == (3, 2)


def test_load_graph_checks_the_manifest(tmp_path):
    folder = write_graph(tmp_path / 'g', ['a\tr\tb', 'b\tr\tc'], ['c\tr\ta'], ['b\tr\ta'])
    manifest = mf.read_manifest(write_manifest(tmp_path / 'm.json', tmp_path, {'g': folder}))
    graph = D.load_graph(manifest, manifest.entry('g'), tmp_path)
    assert graph.num_entities == 3 and graph.hashes['train.txt'] == mf.file_sha256(folder / 'train.txt')
    (folder / 'train.txt').write_text('a\tr\tb\nb\tr\tc\nc\tr\tb\n')
    with pytest.raises(mf.ManifestError, match='train.txt'):
        D.load_graph(manifest, manifest.entry('g'), tmp_path)


@pytest.mark.parametrize('name', ['UMLS', 'NELLInductive_v1', 'grail-WN18RR_v1'])
def test_gate_graphs_give_the_research_rows(name):
    """Sizes, open relations and the row order of the research harness (G6 header and row identity), without a model."""
    gate = mf.read_manifest('gate')
    entry = gate.entry(name)
    if not gate.folder(entry, mf.REPO).is_dir():
        pytest.skip(f'{entry.path} is not available')
    reference = fixture('g6_ranks.json.gz')['graphs'][name]
    graph = D.load_graph(gate, entry, mf.REPO)
    queries = D.prepare_queries(graph, gate.sample_rule)
    header = reference['header']
    assert (graph.num_entities, graph.num_relations, sorted(queries.reflexive)) == (header['entities'], header['relations'], header['reflexive'])
    order = [[queries.index(m), int(head)] for (head, _, _), members in queries.groups for m in members]
    assert len(order) == reference['rows'] and int_digest(order) == reference['rows_identity']
    assert queries.positions is None and queries.test_rows == reference['record']['test_rows']
    assert isinstance(queries.gt2[False, 0][0], np.ndarray)


def test_a_large_test_split_is_sampled_and_keeps_its_filter(tmp_path):
    names = [f'e{i:02d}' for i in range(40)]
    facts = [(h, r, t) for h in names for r in ('r0', 'r1') for t in names if h != t]
    random.Random(1).shuffle(facts)
    train, test = facts[:300], facts[300:1500] + facts[300:320]   # 1,200 distinct test facts, 20 rows repeated
    graph = D.read_graph(write_graph(tmp_path / 'g', ['\t'.join(f) for f in train], [], ['\t'.join(f) for f in test]))
    queries = D.prepare_queries(graph, 'first-1000')
    rows = D.as_triples(graph.test)
    positions = queries.positions
    assert positions is not None and positions == D.sample_positions(rows) and queries.test_rows == 1220 and len(queries.facts) == 1000
    assert all(p < 1200 for p in positions) and queries.facts == [rows[p] for p in positions]
    assert [queries.index(m) for m in range(1000)] == positions
    unsampled = [rows[i] for i in range(1200) if i not in set(positions)]
    assert all(t in queries.known[False][r, h] and h in queries.known[True][r, t] for h, r, t in unsampled)
    members = sorted(m for _, group in queries.groups for m in group)
    assert members == sorted(2 * list(range(1000)))
