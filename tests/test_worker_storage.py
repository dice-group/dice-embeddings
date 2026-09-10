"""Dataset semantics and storage sharing across DataLoader worker start methods."""
import concurrent.futures
import multiprocessing
import pickle
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from dicee.dataset_classes._bpe import MultiLabelDataset
from dicee.dataset_classes._label_based import AllvsAll, FSDP1vsSampleDataset, KvsAll, KvsSampleDataset, OnevsAllDataset
from dicee.dataset_classes._negative_sampling import GroupedNegativeSamplingDataset, TriplePredictionDataset
from dicee.dataset_classes._storage import PairIndex, RaggedIndices
from dicee.read_preprocess_save_load_kg.preprocess import PreprocessKG
from dicee.read_preprocess_save_load_kg.util import get_ee_vocab, get_er_vocab, get_filter_vocabs, get_re_vocab

FACTS = np.array([[0, 0, 2], [2, 1, 4], [0, 0, 1], [1, 1, 0], [4, 2, 5], [2, 1, 3], [0, 0, 2]], dtype=np.int32)


def test_ragged_empty_duplicate_and_ordered_rows():
    rows = [[], [4, 1, 4], [2], []]
    data = RaggedIndices.from_rows(rows)
    assert len(data) == 4
    assert data.max_length == 3
    assert [row.tolist() for row in data] == rows
    assert data[-1].tolist() == []
    with pytest.raises(IndexError):
        data[4]
    assert RaggedIndices.from_rows(data) is data
    assert RaggedIndices.from_rows([]).max_length == 0
    pad, lengths = data.padded_rows(np.array([3, 1, 0]), 9)
    assert pad.tolist() == [[9, 9, 9], [4, 1, 4], [9, 9, 9]]
    assert lengths.tolist() == [0, 3, 0]


@pytest.mark.parametrize('columns', [(0, 1, 2), (0, 2, 1), (1, 2, 0)])
def test_pair_index_preserves_mapping_semantics(columns):
    expected = {}
    for row in FACTS:
        expected.setdefault(tuple(row[list(columns[:2])]), []).append(int(row[columns[2]]))
    index = PairIndex.from_triples(FACTS, columns=columns)
    assert index.keys.tolist() == [list(key) for key in sorted(expected)]
    for key, targets in expected.items():
        assert index[key].tolist() == targets
    unique = PairIndex.from_triples(FACTS, columns=columns, unique=True)
    for key, targets in expected.items():
        assert unique[key].tolist() == sorted(set(targets))
    with pytest.raises(KeyError):
        index[(99, 99)]


def test_pair_lookup_avoids_integer_product_overflow():
    triples = np.array([[2**40, 2**39, 1], [2**40, 2**39 + 1, 2]], dtype=np.int64)
    index = PairIndex.from_triples(triples)
    assert index.find_rows(triples[::-1, :2]).tolist() == [1, 0]
    assert PairIndex.from_triples(np.empty((0, 3), dtype=np.int64)).find_rows([]).size == 0


@pytest.mark.parametrize('kind', ['KvsAll', 'Relation', 'AllvsAll'])
@pytest.mark.parametrize('smoothing', [0., 0.1])
def test_dense_labels_and_pair_order(kind, smoothing):
    if kind == 'AllvsAll':
        dataset = AllvsAll(FACTS, range(7), range(3), label_smoothing_rate=smoothing)
        keys = [(h, r) for h in range(7) for r in range(3)]
    else:
        dataset = KvsAll(FACTS, range(7), range(3),
                         'RelationPrediction' if kind == 'Relation' else 'EntityPrediction',
                         label_smoothing_rate=smoothing)
        columns = [0, 2] if kind == 'Relation' else [0, 1]
        keys = sorted(set(map(tuple, FACTS[:, columns])))
    for i, key in enumerate(keys):
        targets = [r if kind == 'Relation' else t for h, r, t in FACTS
                   if (h, t if kind == 'Relation' else r) == key]
        expected = torch.zeros(3 if kind == 'Relation' else 7)
        expected[targets] = 1
        if smoothing:
            expected = expected * (1 - smoothing) + smoothing / len(expected)
        x, labels = dataset[i]
        assert x.tolist() == list(key)
        assert torch.equal(labels, expected)
    assert isinstance(dataset.train_target, RaggedIndices)


def test_sampled_labels_preserve_positive_order_duplicates_and_negative_rng():
    dataset = KvsSampleDataset(FACTS, range(7), range(3), 'EntityPrediction', neg_ratio=3)
    for i, pair in enumerate(dataset.train_data.tolist()):
        positives = [int(t) for h, r, t in FACTS if [h, r] == pair]
        torch.manual_seed(123)
        weights = torch.ones(7)
        weights[positives] = 0
        negatives = torch.multinomial(weights, dataset.max_num_of_classes - len(positives), replacement=True)
        torch.manual_seed(123)
        x, indices, labels = dataset[i]
        assert indices.tolist() == positives + negatives.tolist()
        assert labels.tolist() == [1.] * len(positives) + [0.] * len(negatives)


def test_bpe_ragged_labels_include_empty_rows():
    targets = [[2, 0, 2], [], [3]]
    dataset = MultiLabelDataset(torch.arange(12).view(3, 2, 2), targets, 4, torch.zeros(4, 2))
    assert isinstance(dataset.train_indices_target, RaggedIndices)
    for i, expected in enumerate([[1., 0., 1., 0.], [0., 0., 0., 0.], [0., 0., 0., 1.]]):
        assert dataset[i][1].tolist() == expected


def test_strict_sampling_matches_seeded_reference_and_excludes_all_positives():
    dataset = GroupedNegativeSamplingDataset(FACTS, 7, 3, neg_sample_ratio=4, strict_negative_sampling=True)
    positives = [dataset[i] for i in range(len(dataset))]
    torch.manual_seed(14)
    triples, labels = dataset.collate_fn(positives)
    generator = torch.Generator().manual_seed(14)
    for i, (h, r, t) in enumerate(torch.stack(positives).tolist()):
        position = 2 if i < len(positives) // 2 else 0
        forbidden = {int(t1 if position == 2 else h1) for h1, r1, t1 in FACTS
                     if (h1 == h and r1 == r if position == 2 else r1 == r and t1 == t)}
        allowed = torch.tensor([v for v in range(7) if v not in forbidden])
        expected = allowed[torch.randint(len(allowed), (4,), generator=generator)]
        assert torch.equal(triples[i, 1:, position], expected)
        assert triples[i, 0].tolist() == [h, r, t]
    assert labels[:, 0].eq(1).all() and labels[:, 1:].eq(0).all()
    assert isinstance(dataset.true_heads, PairIndex) and isinstance(dataset.true_tails, PairIndex)


def test_fsdp_sampling_matches_uniform_allowed_entity_reference():
    dataset = FSDP1vsSampleDataset(FACTS, range(7), range(3), 'EntityPrediction', neg_ratio=3,
                                  label_smoothing_rate=0.1)
    assert dataset.max_num_of_classes == 6  # Three input positives + three negatives, as before.
    np.random.seed(42)
    source, targets, labels = dataset._collate([dataset[i] for i in range(len(dataset))])
    np.random.seed(42)
    draws = np.random.uniform(size=(len(FACTS), dataset.num_negatives))
    for i, (h, r, t) in enumerate(FACTS):
        allowed = [v for v in range(7) if not any((FACTS == [h, r, v]).all(axis=1))]
        expected = np.asarray(allowed)[(draws[i] * len(allowed)).astype(int)]
        assert targets[i].tolist() == [t, *expected]
    assert source.tolist() == FACTS[:, :2].tolist()
    assert torch.allclose(labels[:, 0], torch.full((len(FACTS),), 0.9))
    assert labels[:, 1:].eq(0.1).all()
    assert not hasattr(dataset, '_pair_to_id') and not hasattr(dataset, '_pos_table')


@pytest.mark.parametrize('kind', ['strict', 'fsdp'])
def test_dense_query_rejects_impossible_negatives(kind):
    facts = np.array([[0, 0, 0]])
    if kind == 'strict':
        dataset = GroupedNegativeSamplingDataset(facts, 1, 1, neg_sample_ratio=2, strict_negative_sampling=True)
    else:
        dataset = FSDP1vsSampleDataset(facts, range(1), range(1), 'EntityPrediction', neg_ratio=2)
    with pytest.raises(ValueError, match='No valid negative'):
        dataset.collate_fn([dataset[0]])


class WorkerBundle(Dataset):
    """Exercise real getitem/collate methods and buffer sharing in each worker."""
    def __init__(self, mapped):
        self.datasets = {
            'kvs': KvsAll(FACTS, range(7), range(3), 'EntityPrediction'),
            'relation': KvsAll(FACTS, range(7), range(3), 'RelationPrediction'),
            'all': AllvsAll(FACTS, range(7), range(3)),
            'sample': KvsSampleDataset(FACTS, range(7), range(3), 'EntityPrediction', neg_ratio=3),
            'bpe': MultiLabelDataset(torch.zeros(3, 2, 2, dtype=torch.long), [[2, 1], [], [0]], 7, torch.zeros(7, 2)),
            'strict': GroupedNegativeSamplingDataset(FACTS, 7, 3, neg_sample_ratio=3, strict_negative_sampling=True),
            'fsdp': FSDP1vsSampleDataset(mapped, range(7), range(3), 'EntityPrediction', neg_ratio=3),
            'neg': TriplePredictionDataset(FACTS, 7, 3, neg_sample_ratio=3),
            'one': OnevsAllDataset(FACTS, range(7)),
        }

    def __len__(self):
        return 4

    def __getitem__(self, index):
        batches = {}
        for name, ds in self.datasets.items():
            collate = ds.collate_fn or default_collate
            batches[name] = collate([ds[0], ds[1]])
        shared = []
        for ds in self.datasets.values():
            for value in vars(ds).values():
                if isinstance(value, PairIndex):
                    shared.extend([value.keys.is_shared(), value.targets.values.is_shared(), value.targets.offsets.is_shared()])
                elif isinstance(value, RaggedIndices):
                    shared.extend([value.values.is_shared(), value.offsets.is_shared()])
                elif isinstance(value, np.ndarray) and not isinstance(value, np.memmap):
                    shared.append(isinstance(value.base, torch.Tensor) and value.base.is_shared())
        return batches, torch.tensor(shared), isinstance(self.datasets['fsdp'].train_data, np.memmap)


@pytest.mark.parametrize('context', [m for m in ('fork', 'spawn', 'forkserver') if m in multiprocessing.get_all_start_methods()])
def test_all_dataset_paths_work_with_multiple_workers(tmp_path, context):
    path = tmp_path / 'triples.dat'
    mapped = np.memmap(path, dtype=np.int32, mode='w+', shape=(len(FACTS) + 2, 3), offset=64)
    mapped[1:-1] = FACTS
    mapped.flush()
    dataset = WorkerBundle(mapped[1:-1])  # Exercise a nonzero mapping and slice offset.
    loader = DataLoader(dataset, batch_size=None, num_workers=2, multiprocessing_context=context)
    for batches, shared, is_mapped in loader:
        assert batches['kvs'][0].tolist() == [[0, 0], [1, 1]]
        assert batches['kvs'][1][0].tolist() == [0., 1., 1., 0., 0., 0., 0.]
        assert batches['bpe'][1][1].sum() == 0
        assert batches['fsdp'][0].tolist() == FACTS[:2, :2].tolist()
        assert is_mapped
        if context != 'fork':
            assert shared.all(), shared


def test_numeric_array_pickle_wrapper_reuses_storage():
    dataset = TriplePredictionDataset(FACTS, 7, 3)
    first = dataset.__getstate__()['train_set']
    second = dataset.__getstate__()['train_set']
    assert first is second
    restored = pickle.loads(pickle.dumps(dataset))
    assert np.array_equal(restored.train_set, dataset.train_set)


@pytest.mark.parametrize('strings', [False, True])
def test_streamed_filter_vocabularies_match_existing_format_and_files(tmp_path, strings):
    data = [[str(x) for x in row] for row in FACTS] if strings else FACTS
    expected = (get_er_vocab(data), get_re_vocab(data), get_ee_vocab(data))
    with patch.object(concurrent.futures, 'ProcessPoolExecutor', side_effect=AssertionError('Must not spawn processes')):
        actual = get_filter_vocabs(iter(data), str(tmp_path))
    assert actual == expected
    for name, vocab in zip(('er_vocab', 're_vocab', 'ee_vocab'), expected):
        assert pickle.loads((tmp_path / f'{name}.p').read_bytes()) == vocab
    assert get_filter_vocabs(iter(data)) == expected


@pytest.mark.parametrize('kind', ['numeric', 'pandas', 'polars'])
@pytest.mark.parametrize('missing', ['valid', 'test', None])
def test_preprocessing_streams_each_available_split(kind, missing):
    kg = SimpleNamespace(byte_pair_encoding=kind != 'numeric')
    expected = []
    for name, rows in [('train', [[0, 0, 1]]), ('valid', [[1, 0, 2]]), ('test', [[2, 0, 3]])]:
        if kind != 'numeric':
            rows = [[str(x) for x in row] for row in rows]
        if name == missing:
            value = None
        else:
            expected.extend(rows)
            value = np.array(rows) if kind == 'numeric' else (
                pd.DataFrame(rows) if kind == 'pandas' else pl.DataFrame(rows, orient='row'))
        setattr(kg, ('raw_' if kind != 'numeric' else '') + name + '_set', value)
    assert list(map(list, PreprocessKG(kg)._evaluation_triples())) == expected


@pytest.mark.parametrize('backend', ['pandas', 'polars'])
@pytest.mark.parametrize('missing', ['valid', 'test', None])
def test_real_kg_builds_filters_without_a_process_pool(tmp_path, backend, missing):
    from dicee.knowledge_graph import KG
    directory = tmp_path / 'kg'
    directory.mkdir()
    for name, text in [('train', 'a\tr\tb\na\tr\tc\n'), ('valid', 'b\tr\tc\n'), ('test', 'c\tr\ta\n')]:
        if name != missing:
            (directory / f'{name}.txt').write_text(text)
    with patch.object(concurrent.futures, 'ProcessPoolExecutor', side_effect=AssertionError('Must not spawn processes')):
        kg = KG(dataset_dir=str(directory), eval_model='test', add_reciprocal=False,
                training_technique='NegSample', backend=backend, separator='\t')
    data = [row for split in (kg.train_set, kg.valid_set, kg.test_set) if split is not None for row in split]
    assert kg.er_vocab == get_er_vocab(data)
    assert kg.re_vocab == get_re_vocab(data)
    assert kg.ee_vocab == get_ee_vocab(data)


@pytest.mark.parametrize('name', ['snapshot', 'swag'])
@pytest.mark.parametrize('future', [False, True])
def test_ensemble_callbacks_accept_ready_and_legacy_filters(tmp_path, monkeypatch, name, future):
    import dicee.callbacks as callbacks
    import dicee.weight_averaging as averaging
    vocab = {(0, 0): [1]}
    stored = vocab
    if future:
        stored = concurrent.futures.Future()
        stored.set_result(vocab)
    trainer = SimpleNamespace(dataset=SimpleNamespace(test_set=FACTS, er_vocab=stored), num_training_batches=1)
    model = SimpleNamespace(args={})
    state = SimpleNamespace(snapshot_dir=str(tmp_path), snapshot_loss=[], weighted_ensemble=False, max_num_models=0)

    class ReachedEvaluator(Exception):
        pass

    def evaluate(**kwargs):
        assert kwargs['er_vocab'] is vocab
        raise ReachedEvaluator

    module, cls = (callbacks, callbacks.LRScheduler) if name == 'snapshot' else (averaging, averaging.SWAG)
    monkeypatch.setattr(module, 'evaluate_ensemble_link_prediction_performance', evaluate)
    with pytest.raises(ReachedEvaluator):
        cls.on_fit_end(state, trainer, model)
