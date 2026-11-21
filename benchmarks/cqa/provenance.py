"""Bind locally trained checkpoints to their recorded dataset and run profile."""

from pathlib import Path

from dicee.query_answering._checkpoint import checksum, input_path, read
from dicee.query_answering.datasets import dataset_spec

REQUIRED_INPUTS = {
    'qto': {'id2ent.pkl', 'id2rel.pkl', 'train.txt', 'valid.txt', 'test.txt'},
    'inductive-gnnqe': {'train_graph.txt', 'val_inference.txt', 'test_inference.txt',
                       'train_queries.pkl', 'train_answers_hard.pkl'}
                       | {f'{split}_{kind}.pkl' for split in ('valid', 'test')
                          for kind in ('queries', 'answers_easy', 'answers_hard')},
}


def validate_training(entry, input_root, data_root):
    """Return the training record of a locally trained checkpoint after checking it against the checkpoint and dataset."""
    path = entry.get('training')
    if not path:
        if (entry['method'] == 'inductive-gnnqe'
                or (entry['method'] == 'qto' and entry['dataset'] == 'FB15kLogicalQuery')):
            raise ValueError('Training provenance is required for this locally trained baseline')
        return None

    def digest(relative):
        return checksum(input_path(input_root, relative))

    record = read(input_path(input_root, path))
    if (not isinstance(record, dict) or record.get('version') != 1 or record.get('status') != 'complete'
            or record.get('method') != entry['method'] or record.get('dataset') != entry['dataset']
            or record.get('checkpoint_sha256') != digest(entry['checkpoint'])):
        raise ValueError('Training provenance does not match the completed checkpoint')
    if record.get('profile') not in ('author-recipe', 'custom', 'smoke', 'fixture'):
        raise ValueError('Training provenance requires a recognized training profile')
    releases = {'qto': ('BetaE', 'fixture'), 'inductive-gnnqe': ('InductiveQE v2.0', 'fixture')}
    if record.get('dataset_release') not in releases.get(entry['method'], ()):
        raise ValueError('Training provenance requires the supported dataset release')
    inputs = record.get('inputs')
    required = REQUIRED_INPUTS.get(entry['method'])
    if not required or not isinstance(inputs, dict) or required - inputs.keys():
        raise ValueError('Training provenance is missing required dataset input bindings')
    prefix = Path(data_root) / dataset_spec(entry['dataset'])[1]
    for name, expected in inputs.items():
        if not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..'):
            raise ValueError('Training inputs must name files within the dataset folder')
        relative = str(prefix / name)
        try:
            actual = digest(relative)
        except FileNotFoundError as error:
            raise ValueError(f'Training input is missing: {relative}') from error
        if actual != expected:
            raise ValueError(f'Training input differs from the evaluation dataset: {relative}')
    return record
