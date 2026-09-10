"""ULTRA contract, graph lifecycle, and reusable negative-sampling tests."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dicee.config import Namespace
from dicee.dataset_classes._negative_sampling import GroupedNegativeSamplingDataset, TriplePredictionDataset
from dicee.models.real import DistMult
from dicee.models.sampled_loss import grouped_adversarial_bce
from dicee.models.ultra import ULTRA


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def facts():
    return torch.tensor([[0, 0, 1], [0, 0, 2], [1, 1, 2], [2, 0, 3], [3, 1, 0]])


def make_model(facts, **kwargs):
    args = dict(num_entities=5, num_relations=2, ultra_dim=8, ultra_num_layers=2, optim='Adam', **kwargs)
    return ULTRA(args).set_graph(facts)


def test_official_architecture():
    model = ULTRA({})
    assert len(model.state_dict()) == 82
    assert sum(p.numel() for p in model.parameters()) == 168705
    assert all(k.startswith(('relation_model.', 'entity_model.')) for k in model.state_dict())
    with pytest.raises(RuntimeError, match='Attach'):
        model(torch.tensor([[0, 0, 1]]))


def test_scoring_and_chunking(facts):
    model = make_model(facts).eval()
    queries = facts[:, :2]
    all_scores = model(queries)
    torch.testing.assert_close(model(facts), all_scores.gather(1, facts[:, 2:]).flatten())
    sampled = model((queries, facts[:, 2:]))
    torch.testing.assert_close(sampled, all_scores.gather(1, facts[:, 2:]))
    grouped = facts[:, None].repeat(1, 5, 1)
    grouped[:, :, 2] = torch.arange(5)
    torch.testing.assert_close(model(grouped), all_scores)
    heads = model.forward_k_vs_all_heads(facts[:, 1:])
    assert heads.shape == (5, 5)
    model.query_batch_size = 1
    torch.testing.assert_close(model(queries), all_scores)
    torch.testing.assert_close(model.forward_k_vs_all_heads(facts[:, 1:]), heads)
    assert torch.isfinite(all_scores).all()  # includes isolated entity 4


def test_graph_permutation_duplicates_and_restore(facts, tmp_path):
    model = make_model(facts).eval()
    before = model(facts)
    # External relation IDs interleave direct and inverse relations.
    permuted = facts.clone()
    permuted[:, 1] = torch.tensor([3, 1])[facts[:, 1]]
    inverse = permuted[:, [2, 1, 0]].clone()
    inverse[:, 1] = torch.tensor([0, 2])[facts[:, 1]]
    model.set_graph(torch.cat((permuted, inverse, permuted)), 5, 4, {3: 0, 1: 2})
    torch.testing.assert_close(model(permuted), before)
    assert model.edge_type.numel() == 2 * len(facts)
    model.save_graph(tmp_path / 'graph.pt')
    restored = ULTRA(dict(num_entities=5, num_relations=4, ultra_dim=8, ultra_num_layers=2)).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.load_graph(tmp_path / 'graph.pt')
    torch.testing.assert_close(restored(permuted), before)
    with pytest.raises(ValueError, match='nonempty'):
        model.set_graph(torch.empty(0, 3), 5, 2)
    with pytest.raises(ValueError, match='vocabulary'):
        model.set_graph(torch.tensor([[0, 2, 1]]), 5, 2)
    with pytest.raises(ValueError, match='disjoint'):
        model.set_graph(facts, 5, 2, {0: 1, 1: 0})


def test_training_edge_removal_and_gradients(facts):
    model = make_model(facts)
    original_edges = model.edge_index.clone()
    original_rel_edges = model.rel_edge_index.clone()
    converted = model._convert(facts[:1])
    edge, typ = model._training_edges(targets=converted)
    remaining = torch.stack((edge[0], typ, edge[1]), 1).tolist()
    assert [0, 0, 1] not in remaining and [1, 2, 0] not in remaining
    assert [0, 0, 2] in remaining
    edge, typ = model._training_edges(queries=converted[:, :2])
    remaining = torch.stack((edge[0], typ, edge[1]), 1).tolist()
    assert [0, 0, 2] not in remaining and [2, 2, 0] not in remaining
    loss = model(facts).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    initial = model.entity_model.mlp[2].weight.clone()
    torch.optim.Adam(model.parameters()).step()
    assert not torch.equal(initial, model.entity_model.mlp[2].weight)
    torch.testing.assert_close(model.edge_index, original_edges)
    torch.testing.assert_close(model.rel_edge_index, original_rel_edges)
    # Held-out facts have never been attached to either graph.
    assert [4, 0, 0] not in model.graph_triples.tolist()


def test_training_chunk_invariance(facts):
    model = make_model(facts)
    first = model(facts[:, :2])
    model.query_batch_size = 1
    torch.testing.assert_close(model(facts[:, :2]), first)


def test_checkpoint_loading(facts, tmp_path):
    model = make_model(facts)
    for wrapped in (False, True):
        path = tmp_path / 'checkpoint.pth'
        torch.save({'model': model.state_dict(), 'optimizer': {}} if wrapped else model.state_dict(), path)
        restored = make_model(facts).load_pretrained(path)
        for k, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[k], value)
    with pytest.raises(ValueError, match='architecture mismatch'):
        ULTRA({}).load_pretrained(path)
    with pytest.raises(FileNotFoundError, match='graph artifact'):
        model.load_graph(tmp_path / 'absent.pt')


def test_reusable_sampler_and_loss(facts):
    dataset = GroupedNegativeSamplingDataset(
        train_set=facts.numpy(), num_entities=5, num_relations=2,
        neg_sample_ratio=7, strict_negative_sampling=True)
    triples, targets = dataset.collate_fn([dataset[i] for i in range(len(dataset))])
    assert triples.shape == (5, 8, 3)
    known = set(map(tuple, facts.tolist()))
    assert all(tuple(t) not in known for t in triples[:, 1:].reshape(-1, 3).tolist())
    assert targets[:, 0].eq(1).all() and targets[:, 1:].eq(0).all()
    model = DistMult(dict(num_entities=5, num_relations=2, embedding_dim=8,
                         adversarial_temperature=1.0, scoring_technique='NegSample'))
    scores = model(triples)
    torch.testing.assert_close(scores, model(triples.reshape(-1, 3)).reshape(5, 8))
    loss = model.training_step((triples, targets))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.entity_embeddings.weight.grad is not None
    dense = GroupedNegativeSamplingDataset(train_set=np.array([[0, 0, 0]]), num_entities=1,
                                          num_relations=1, neg_sample_ratio=1, strict_negative_sampling=True)
    with pytest.raises(ValueError, match='No valid negative'):
        dense.collate_fn([dense[0]])


@pytest.mark.parametrize('temperature', [0, 0.5, 1])
def test_adversarial_loss_formula(temperature):
    logits = torch.tensor([[1., -1., 2.], [-2., 0., 1.]], requires_grad=True)
    labels = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
    loss = grouped_adversarial_bce(logits, labels, temperature)
    weights = torch.softmax(logits[:, 1:].detach() / temperature, -1) if temperature else torch.full((2, 2), 0.5)
    expected = (torch.nn.functional.softplus(-logits[:, 0]) + (weights * torch.nn.functional.softplus(logits[:, 1:])).sum(-1)).mean() / 2
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(torch.autograd.grad(loss, logits)[0], torch.autograd.grad(expected, logits)[0])


def test_original_sampler_default(facts):
    dataset = TriplePredictionDataset(facts.numpy(), 5, 2, neg_sample_ratio=2)
    x, y = dataset.collate_fn([dataset[0], dataset[1]])
    assert x.shape == (6, 3) and y.shape == (6,)


@pytest.mark.parametrize('technique,grouped,trainer,backend', [
    ('NegSample', False, 'torchCPUTrainer', 'pandas'),
    ('NegSample', True, 'torchCPUTrainer', 'pandas'),
    ('KvsAll', False, 'torchCPUTrainer', 'pandas'),
    ('NegSample', True, 'PL', 'pandas'),
    ('KvsAll', False, 'torchCPUTrainer', 'polars'),
    ('KvsSample', False, 'torchCPUTrainer', 'pandas'),
    ('1vsSample', False, 'torchCPUTrainer', 'pandas'),
    ('1vsAll', False, 'torchCPUTrainer', 'pandas'),
    ('FixedNegSample', False, 'torchCPUTrainer', 'pandas'),
])
def test_execute_and_reload(tmp_path, technique, grouped, trainer, backend):
    from dicee.executer import Execute
    from dicee.knowledge_graph_embeddings import KGE
    dataset = tmp_path / 'data'
    dataset.mkdir()
    (dataset / 'train.txt').write_text('a r b\na r c\nb s c\nc r d\nd s a\n')
    (dataset / 'valid.txt').write_text('b r d\n')
    (dataset / 'test.txt').write_text('d r b\n')
    args = Namespace()
    args.model, args.ultra_dim, args.ultra_num_layers = 'ULTRA', 8, 2
    args.dataset_dir, args.path_to_store_single_run = str(dataset), str(tmp_path / 'run')
    args.num_epochs, args.batch_size, args.neg_ratio, args.lr = 1, 2, 2, 0.001
    args.scoring_technique, args.trainer = technique, trainer
    args.backend, args.separator = backend, " "
    args.strict_negative_sampling = grouped
    args.adversarial_temperature = 1.0 if grouped else None
    if trainer == 'PL':
        args.pl_trainer_kwargs = {'accelerator': 'cpu', 'devices': 1}
    execute = Execute(args)
    report = execute.start()
    assert np.isfinite(report['Test']['MRR'])
    restored = KGE(path=str(tmp_path / 'run'))
    triples = torch.tensor([[0, 0, 1]])
    torch.testing.assert_close(restored.model(triples), execute.trained_model(triples))
    assert (tmp_path / 'run' / 'ultra_graph.pt').is_file()
    restored.predict_missing_head_entity('r', 'b')
    restored.predict_missing_tail_entity('a', 'r')
    ensemble = KGE(path=str(tmp_path / 'run'), construct_ensemble=True)
    torch.testing.assert_close(ensemble.model(triples), restored.model(triples))


def test_zero_epoch_checkpoint(tmp_path):
    from dicee.executer import Execute
    dataset = tmp_path / 'data'
    dataset.mkdir()
    (dataset / 'train.txt').write_text('a r b\nb r c\nc r a\n')
    (dataset / 'test.txt').write_text('a r c\n')
    path = tmp_path / 'pretrained.pth'
    pretrained = ULTRA({})
    torch.save({'model': pretrained.state_dict()}, path)
    args = Namespace()
    args.model, args.ultra_checkpoint = 'ULTRA', str(path)
    args.dataset_dir, args.path_to_store_single_run = str(dataset), str(tmp_path / 'run')
    args.num_epochs, args.batch_size = 0, 2
    args.scoring_technique, args.eval_model = 'NegSample', 'test'
    report = Execute(args).start()
    state = torch.load(tmp_path / 'run' / 'model.pt', weights_only=True)
    for key in state:
        torch.testing.assert_close(state[key], pretrained.state_dict()[key])
    assert np.isfinite(report['Test']['MRR'])
    assert json.loads((tmp_path / 'run' / 'configuration.json').read_text())['ultra_dim'] == 64


def test_reciprocal_head_conditioning(facts):
    model = make_model(facts).eval()
    before = model.forward_k_vs_all_heads(facts[:, 1:])
    inverse = facts[:, [2, 1, 0]].clone()
    inverse[:, 1] += 2
    model.set_graph(torch.cat((facts, inverse)), 5, 4, {0: 2, 1: 3})
    torch.testing.assert_close(model(inverse[:, :2]), before)
    torch.testing.assert_close(model(inverse), before.gather(1, facts[:, :1]).flatten())


@pytest.mark.parametrize('name', ['tiny', 'ultra_3g', 'ultra_4g', 'ultra_50g'])
def test_upstream_numerical_parity(name):
    import hashlib
    import os
    fixture = torch.load(Path(__file__).parent / 'fixtures' / 'ultra' / (name + '.pt'), weights_only=True)
    model = ULTRA(dict(num_entities=fixture['num_entities'], num_relations=fixture['num_relations'],
                       ultra_dim=fixture['dim'], ultra_num_layers=fixture['num_layers']))
    if name == 'tiny':
        model.load_state_dict(fixture['state_dict'], strict=True)
    else:
        directory = os.environ.get('ULTRA_CHECKPOINT_DIR')
        if directory is None:
            pytest.skip('Set ULTRA_CHECKPOINT_DIR to the official ckpts directory')
        path = Path(directory) / (name + '.pth')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture['checkpoint_sha256']
        model.load_pretrained(path)
    model.set_graph(fixture['triples']).eval()
    actual_edges = torch.cat((model.rel_edge_index.T, model.rel_edge_type[:, None]), 1)
    expected_edges = torch.cat((fixture['rel_edge_index'].T, fixture['rel_edge_type'][:, None]), 1)
    assert set(map(tuple, actual_edges.tolist())) == set(map(tuple, expected_edges.tolist()))
    torch.testing.assert_close(model.relation_model(model.rel_edge_index, model.rel_edge_type, 6, fixture['queries'][:, 1]),
                               fixture['relations'], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(model(fixture['queries'][:, :2]), fixture['tails'], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(model.forward_k_vs_all_heads(fixture['queries'][:, 1:]), fixture['heads'], atol=1e-5, rtol=1e-4)
    model.train()
    training_scores = model(fixture['grouped'])
    torch.testing.assert_close(training_scores, fixture['training_scores'], atol=1e-5, rtol=1e-4)
    if name == 'tiny':
        training_scores.sum().backward()
        for key, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, fixture['gradients'][key], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('overrides,message', [
    ({'trainer': 'torchFSDP'}, 'CPU/single GPU'),
    ({'byte_pair_encoding': True}, 'BPE'),
    ({'num_folds_for_cv': 3}, 'cross-validation'),
    ({'save_embeddings_as_csv': True}, 'embedding export'),
    ({'pl_trainer_kwargs': {'devices': 2}}, 'single device'),
    ({'normalization': 'LayerNorm'}, 'internal layer normalization'),
])
def test_unsupported_ultra_settings(tmp_path, overrides, message):
    from dicee.static_preprocess_funcs import preprocesses_input_args
    args = Namespace()
    args.model, args.dataset_dir = 'ULTRA', str(tmp_path)
    for key, value in overrides.items():
        setattr(args, key, value)
    with pytest.raises(ValueError, match=message):
        preprocesses_input_args(args)


def test_memmap_graph_attachment_and_replacement(facts, tmp_path):
    import pandas as pd

    from dicee.trainer.dice_trainer import DICE_Trainer
    path = tmp_path / 'facts.mmap'
    data = np.memmap(path, mode='w+', dtype='int64', shape=facts.shape)
    data[:] = facts.numpy()
    data.flush()
    pd.DataFrame({'relation': ['r', 's']}).to_csv(tmp_path / 'relation_to_idx.csv')
    args = Namespace()
    args.model = 'ULTRA'
    args.num_entities, args.num_relations = 5, 2
    args.apply_reciprical_or_noise = False
    trainer = DICE_Trainer(args, False, str(tmp_path), None)
    model = make_model(facts).eval()
    trainer.prepare_ultra(model, data)
    before = model(facts)
    model.set_graph(torch.tensor([[4, 0, 0], [0, 1, 2]]))
    assert model.edge_type.numel() == 4
    assert not torch.allclose(model(facts), before)
    trainer.prepare_ultra(model, data)
    torch.testing.assert_close(model(facts), before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_cuda_graph_movement(facts):
    model = make_model(facts).eval()
    before = model(facts)
    model.cuda()
    assert model.graph_triples.is_cuda and model.rel_edge_index.is_cuda
    torch.testing.assert_close(model(facts).cpu(), before, atol=1e-5, rtol=1e-4)
    model.train()
    model(facts).sum().backward()
    assert all(p.grad is not None and p.grad.is_cuda for p in model.parameters())
