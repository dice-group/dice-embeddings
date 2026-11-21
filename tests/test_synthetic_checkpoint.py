"""The checkpoint format and its loader (``dicee.synthetic_kg.checkpoint``)."""
import pytest
import torch

from dicee.models.ultra import ULTRA
from dicee.synthetic_kg.checkpoint import (
    FORMAT,
    KEYS,
    CheckpointError,
    atomic_write_bytes,
    check_checkpoint,
    load_model,
    make_checkpoint,
    read_checkpoint,
    save_checkpoint,
)
from dicee.synthetic_kg.determinism import THREADS, torch_threads
from dicee.synthetic_kg.digests import tensor_digest
from dicee.synthetic_kg.variants import VARIANTS, initial_model


@pytest.fixture(autouse=True)
def two_threads():
    with torch_threads(THREADS):
        yield


def checkpoint_of(variant: str, seed: int = 42, **changes) -> dict:
    model = initial_model(variant, seed)
    values = dict(variant=variant, seed=seed, updates=0, weights='raw', order_sha256=VARIANTS[variant].pinned_order(seed), provenance={'source': 'test'})
    return make_checkpoint(model.state_dict(), **{**values, **changes})


def test_checkpoint_round_trip(tmp_path):
    checkpoint = checkpoint_of('PCUltra-S')
    path = save_checkpoint(checkpoint, tmp_path / 'PCUltra-S-42.pt')
    stored = torch.load(path, map_location='cpu', weights_only=True)
    assert set(stored) == set(KEYS) and stored['format'] == FORMAT and stored['model_config'] == VARIANTS['PCUltra-S'].model_config()
    assert stored['variant'] == 'PCUltra-S' and stored['provenance'] == {'source': 'test'}
    assert tensor_digest(read_checkpoint(path)['model']) == tensor_digest(checkpoint['model'])
    before = torch.get_rng_state()
    model = load_model(path)
    assert torch.equal(torch.get_rng_state(), before)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert (model.relation_weighting, model.pair_channels) == ('normalized', 'on')
    assert tensor_digest(model.state_dict()) == tensor_digest(checkpoint['model'])
    direct = ULTRA(dict(checkpoint['model_config'])).load_pretrained(path)   # the plain dicee loader reads the format
    assert tensor_digest(direct.state_dict()) == tensor_digest(checkpoint['model'])
    with pytest.raises(ValueError, match='saved with'):
        ULTRA(VARIANTS['ULTRA-S'].model_config()).load_pretrained(path)


@pytest.mark.parametrize('variant', ['CountULTRA-R', 'ULTRA-M'])
def test_checkpoints_of_the_other_models_load_with_their_options(variant, tmp_path):
    model = load_model(save_checkpoint(checkpoint_of(variant, 43), tmp_path / 'checkpoint.pt'))
    assert (model.relation_weighting, model.pair_channels) == (VARIANTS[variant].relation_weighting, 'off')
    assert tensor_digest(model.state_dict()) == tensor_digest(initial_model(variant, 43).state_dict())


def test_a_failed_atomic_write_leaves_no_temporary_file(tmp_path):
    (tmp_path / 'target').mkdir()   # os.replace cannot put a file in place of a folder that holds a file
    (tmp_path / 'target' / 'kept').touch()
    with pytest.raises(OSError):
        atomic_write_bytes(tmp_path / 'target', b'data')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['target']


def test_checkpoint_checks(tmp_path):
    with pytest.raises(CheckpointError, match='keys and shapes'):
        make_checkpoint(initial_model('PCUltra-S', 42).state_dict(), variant='CountULTRA-S', seed=42, updates=0, weights='raw',
                        order_sha256=str(VARIANTS['CountULTRA-S'].pinned_order(42)), provenance={})
    with pytest.raises(CheckpointError, match='order_sha256'):
        checkpoint_of('PCUltra-S', order_sha256=VARIANTS['PCUltra-M'].pinned_order(42))
    with pytest.raises(CheckpointError, match='weights'):
        checkpoint_of('ULTRA-S', weights='swa')
    checkpoint_of('ULTRA-S', seed=7, order_sha256='0' * 64)   # an unpinned seed takes any order digest
    good = checkpoint_of('CountULTRA-S')
    for change in (dict(format='other'), dict(variant='S'), dict(model_config=VARIANTS['PCUltra-S'].model_config()), dict(seed=-1)):
        with pytest.raises(CheckpointError):
            check_checkpoint({**good, **change})
    with pytest.raises(CheckpointError, match='keys'):
        check_checkpoint({k: v for k, v in good.items() if k != 'provenance'})
    with pytest.raises(CheckpointError, match='keys'):   # a file with the field name of an earlier draft of the format
        check_checkpoint({('arm' if k == 'variant' else k): v for k, v in good.items()})
    torch.save({**good, 'format': 'other'}, tmp_path / 'bad.pt')
    with pytest.raises(CheckpointError, match='format'):
        load_model(tmp_path / 'bad.pt')
    (tmp_path / 'text.pt').write_text('not a checkpoint')
    (tmp_path / 'empty.pt').write_bytes(b'')
    for name, message in (('text.pt', 'does not load'), ('empty.pt', 'does not load'), ('missing.pt', 'does not exist')):
        for reader in (read_checkpoint, load_model):   # every load error is a CheckpointError
            with pytest.raises(CheckpointError, match=f'{name} {message}'):
                reader(tmp_path / name)
    state = initial_model('PCUltra-S', 42).state_dict()
    next(iter(state.values())).view(-1)[0] = float('nan')
    with pytest.raises(CheckpointError, match='not finite'):
        make_checkpoint(state, variant='PCUltra-S', seed=42, updates=0, weights='raw',
                        order_sha256=str(VARIANTS['PCUltra-S'].pinned_order(42)), provenance={})
