"""ReadFromDisk routes the split files of a dataset folder by their file names."""

from types import SimpleNamespace

import pytest

from dicee.read_preprocess_save_load_kg.read_from_disk import ReadFromDisk

TRAIN = [('a', 'r', 'b'), ('b', 'r', 'c')]
VALID = [('a', 'r', 'c')]
TEST = [('c', 'r', 'a')]


def _write(path, triples) -> None:
    path.write_text(''.join(f'{head}\t{relation}\t{tail}\n' for head, relation, tail in triples))


def _rows(frame) -> list:
    return [tuple(row) for row in frame.values.tolist()]


@pytest.mark.parametrize('folder', ['train-valid-test-kg', 'test', 'valid'])
def test_split_files_are_routed_by_file_name(tmp_path, folder):
    """A folder name that contains a split word must not decide the split of its files."""
    dataset = tmp_path / folder
    dataset.mkdir()
    _write(dataset / 'train.txt', TRAIN)
    _write(dataset / 'valid.txt', VALID)
    _write(dataset / 'test.txt', TEST)
    kg = SimpleNamespace(path_single_kg=None, sparql_endpoint=None, dataset_dir=str(dataset), read_only_few=None,
                         sample_triples_ratio=None, backend='pandas', separator=r'\s+', add_noise_rate=None,
                         eval_model='train_val_test')

    ReadFromDisk(kg).start()

    assert _rows(kg.raw_train_set) == TRAIN
    assert _rows(kg.raw_valid_set) == VALID
    assert _rows(kg.raw_test_set) == TEST
