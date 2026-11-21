"""PCUltra equals ULTRA with count weighting and pair channels: relation graph, weights, parameters, scores and checkpoints."""
import pytest
import torch

from dicee.models.pcultra import EDGE_TYPES, PCUltra, PCUltraRelationModel, count_weights, pair_edges, sparse_relation_graph
from dicee.models.ultra import ULTRA, relation_graph
from dicee.synthetic_kg.determinism import THREADS, torch_threads

CONFIG = {'ultra_dim': 16, 'ultra_num_layers': 3, 'ultra_query_batch_size': 4}
OPTIONS = {'ultra_relation_weighting': 'normalized', 'ultra_pair_channels': 'on'}


@pytest.fixture(autouse=True)
def two_threads():
    with torch_threads(THREADS):
        yield


def random_graph(seed: int, entities: int = 40, relations: int = 6, facts: int = 160) -> tuple[torch.Tensor, int, int]:
    """Random facts with repeated facts, reversed facts and a self-loop."""
    g = torch.Generator().manual_seed(seed)
    triples = torch.stack((torch.randint(entities, (facts,), generator=g), torch.randint(relations, (facts,), generator=g),
                           torch.randint(entities, (facts,), generator=g)), 1)
    triples = torch.cat((triples, triples[:5], triples[:3, [2, 1, 0]], torch.tensor([[1, 0, 1]])))
    return triples, entities, relations


def option_form(seed: int = 0) -> ULTRA:
    torch.manual_seed(seed)
    return ULTRA({**CONFIG, **OPTIONS})


def pcultra(seed: int = 0, **config) -> PCUltra:
    torch.manual_seed(seed)
    return PCUltra({**CONFIG, **config})


def frozen(model):
    return model.eval().requires_grad_(False)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_relation_graph_and_weights_equal_the_option_form(seed):
    triples, n, r = random_graph(seed)
    old, new = option_form().set_graph(triples, n, r), pcultra().set_graph(triples, n, r)
    assert torch.equal(old.rel_edge_index, new.rel_edge_index) and torch.equal(old.rel_edge_type, new.rel_edge_type)
    assert torch.equal(old.rel_edge_weight, new.rel_edge_weight) and set(new.rel_edge_type.tolist()) == set(range(EDGE_TYPES))
    dense = relation_graph(new.graph_triples, n, r, 'normalized', 'on')
    sparse = sparse_relation_graph(new.graph_triples, n, r)
    assert sparse.pair_channels == 'on' and sparse.edge_weight is not None and sparse.edge_weight.dtype == torch.float64
    assert torch.equal(dense.edge_index, sparse.edge_index) and torch.equal(dense.edge_type, sparse.edge_type)
    assert torch.equal(dense.edge_weight, sparse.edge_weight)   # bitwise, in float64


def test_relation_graphs_without_pair_edges_and_without_facts():
    chain = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 2, 3]])   # no entity pair carries two relations
    model = pcultra().set_graph(chain, 4, 3)
    dense = relation_graph(model.graph_triples, 4, 3, 'normalized', 'on')
    assert int((model.rel_edge_type >= 4).sum()) == 0 and torch.equal(dense.edge_index, model.rel_edge_index)
    assert torch.equal(dense.edge_weight.float(), model.rel_edge_weight)
    edges, kinds, weights = pair_edges(torch.zeros((0, 3), dtype=torch.long), 4, 3)
    assert edges.shape == (2, 0) and len(kinds) == len(weights) == 0
    with pytest.raises(ValueError, match='without a shared entity'):   # an edge that the facts do not give
        count_weights(chain, 4, 3, torch.tensor([[0], [2]]), torch.tensor([0]))


def test_parameters_equal_the_option_form():
    old, new = option_form(7), pcultra(7)
    assert list(new.state_dict()) == list(old.state_dict()) and isinstance(new.relation_model, PCUltraRelationModel)
    assert all(torch.equal(value, old.state_dict()[key]) for key, value in new.state_dict().items())
    tables = [k for k in new.state_dict() if k.startswith('relation_model.layers.') and k.endswith('.relation.weight')]
    assert len(tables) == CONFIG['ultra_num_layers'] and all(new.state_dict()[k].shape == (EDGE_TYPES, CONFIG['ultra_dim']) for k in tables)
    full = PCUltra({})
    assert len(full.state_dict()) == 82 and (full.name, full.graph_filename, full.config_prefix) == ('PCUltra', 'pcultra_graph.pt', 'ultra')


def test_the_two_options_are_fixed():
    model = PCUltra({**CONFIG, 'ultra_relation_weighting': 'binary', 'ultra_pair_channels': 'off'})
    assert (model.relation_weighting, model.pair_channels) == ('normalized', 'on')


@pytest.mark.parametrize('seed', [0, 3])
def test_sparse_product_matches_the_weighted_layer(seed):
    triples, n, r = random_graph(seed)
    model = frozen(pcultra(seed).set_graph(triples, n, r))
    reference = frozen(option_form(seed).set_graph(triples, n, r))
    query = torch.arange(2 * r)
    arguments = (model.rel_edge_index, model.rel_edge_type, 2 * r, query)
    with torch.no_grad():
        fast = model.relation_model(*arguments, edge_weight=model.rel_edge_weight)
        expected = reference.relation_model(*arguments, edge_weight=reference.rel_edge_weight)
        torch.testing.assert_close(fast, expected, rtol=1e-5, atol=1e-5)
        model.set_inference_backend('torch')
        assert torch.equal(model.relation_model(*arguments, edge_weight=model.rel_edge_weight), expected)   # the same operations


@pytest.mark.parametrize('seed', [5, 6])
def test_scores_and_ranks_equal_the_option_form(seed):
    triples, n, r = random_graph(seed)
    old, new = frozen(option_form(11)), frozen(pcultra(0))
    new.load_state_dict(old.state_dict())
    for model in (old, new):
        model.set_graph(triples, n, r)
    x = triples[:12, :2]
    with torch.no_grad():
        a, b = new.forward_k_vs_all(x), old.forward_k_vs_all(x)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(new.forward_k_vs_all_heads(triples[:12, 1:]), old.forward_k_vs_all_heads(triples[:12, 1:]), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(new(triples[:20]), old(triples[:20]), rtol=1e-5, atol=1e-5)
        new.set_inference_backend('torch')
        assert torch.equal(new.forward_k_vs_all(x), b)


def test_training_uses_the_weighted_layer():
    triples, n, r = random_graph(8)
    old, new = option_form(2).set_graph(triples, n, r), pcultra(2).set_graph(triples, n, r)
    old.train()
    new.train()
    a, b = new(triples[:16]), old(triples[:16])
    assert torch.equal(a, b)
    a.sum().backward()
    b.sum().backward()
    gradients = dict(old.named_parameters())
    assert all(p.grad is not None and torch.equal(p.grad, gradients[name].grad) for name, p in new.named_parameters())


def test_scoring_needs_the_attached_graph():
    triples, n, r = random_graph(0)
    model = frozen(pcultra())
    with pytest.raises(RuntimeError, match='set_graph'):
        model.forward_k_vs_all(triples[:2, :2])
    model.set_graph(triples, n, r)
    with pytest.raises(RuntimeError, match='Attach the weighted relation graph'), torch.no_grad():
        model.relation_model(model.rel_edge_index, model.rel_edge_type, 2 * r, torch.arange(r))
    with pytest.raises(ValueError, match='repeated edges'):
        model.relation_model.attach(torch.zeros((2, 2), dtype=torch.long), torch.zeros(2, dtype=torch.long), torch.ones(2), 2 * r)


def test_a_new_graph_replaces_the_matrix():
    triples, n, r = random_graph(1)
    other, m, s = random_graph(2, entities=25, relations=4, facts=90)
    model, reference = frozen(pcultra(4)), frozen(option_form(4))
    with torch.no_grad():
        for facts, entities, relations in ((triples, n, r), (other, m, s), (triples, n, r)):
            model.set_graph(facts, entities, relations)
            reference.set_graph(facts, entities, relations)
            torch.testing.assert_close(model.forward_k_vs_all(facts[:6, :2]), reference.forward_k_vs_all(facts[:6, :2]), rtol=1e-5, atol=1e-5)
    assert model.relation_model.matrix_values.numel() == model.rel_edge_type.numel()


def test_checkpoints_of_the_synthetic_prior(tmp_path):
    from dicee.synthetic_kg.checkpoint import load_model, make_checkpoint, save_checkpoint
    from dicee.synthetic_kg.variants import VARIANTS, initial_model
    triples, n, r = random_graph(9)
    paths = {}
    for variant in ('PCUltra-S', 'CountULTRA-S'):
        checkpoint = make_checkpoint(initial_model(variant, 42).state_dict(), variant=variant, seed=42, updates=0, weights='raw',
                                     order_sha256=VARIANTS[variant].pinned_order(42), provenance={'source': 'test'})
        paths[variant] = save_checkpoint(checkpoint, tmp_path / f'{variant}.pt')
    model = frozen(PCUltra({}).load_pretrained(paths['PCUltra-S']).set_graph(triples, n, r))
    reference = load_model(paths['PCUltra-S']).set_graph(triples, n, r)
    with torch.no_grad():
        torch.testing.assert_close(model.forward_k_vs_all(triples[:8, :2]), reference.forward_k_vs_all(triples[:8, :2]), rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="saved with {'ultra_pair_channels': 'off'}"):   # CountULTRA has no pair channels
        PCUltra({}).load_pretrained(paths['CountULTRA-S'])


def test_saved_graph_restores_the_matrix(tmp_path):
    triples, n, r = random_graph(3)
    model = frozen(pcultra(1).set_graph(triples, n, r))
    model.save_graph(tmp_path / model.graph_filename)
    loaded = frozen(pcultra(1, num_entities=n, num_relations=r))
    loaded.load_graph(tmp_path / model.graph_filename)
    with torch.no_grad():
        assert torch.equal(loaded.forward_k_vs_all(triples[:6, :2]), model.forward_k_vs_all(triples[:6, :2]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_cuda_sparse_product_matches_the_weighted_layer():
    triples, n, r = random_graph(4, entities=300, relations=12, facts=2000)
    model = frozen(pcultra(4).set_graph(triples, n, r).to('cuda'))
    query = torch.arange(2 * r, device='cuda')
    arguments = (model.rel_edge_index, model.rel_edge_type, 2 * r, query)
    with torch.no_grad():
        fast = model.relation_model(*arguments, edge_weight=model.rel_edge_weight)
        model.set_inference_backend('torch')
        plain = model.relation_model(*arguments, edge_weight=model.rel_edge_weight)
    torch.testing.assert_close(fast, plain, rtol=1e-5, atol=1e-5)


def test_the_framework_knows_the_model():
    from dicee.scripts.run import get_default_arguments
    from dicee.static_funcs import MODEL_REGISTRY
    assert MODEL_REGISTRY['PCUltra'] == (PCUltra, 'EntityPrediction')
    assert get_default_arguments(['--model', 'PCUltra']).model == 'PCUltra'


def test_zero_shot_run_and_reload(tmp_path):
    """A checkpoint of the synthetic prior through the framework: the evaluation, the stored experiment and its reload."""
    from dicee.config import Namespace
    from dicee.executer import Execute
    from dicee.knowledge_graph_embeddings import KGE
    from dicee.synthetic_kg.checkpoint import make_checkpoint, save_checkpoint
    from dicee.synthetic_kg.variants import VARIANTS, initial_model
    dataset = tmp_path / 'data'
    dataset.mkdir()
    (dataset / 'train.txt').write_text('a r b\na r c\nb s c\nc r d\nd s a\na s c\n')
    (dataset / 'valid.txt').write_text('b r d\n')
    (dataset / 'test.txt').write_text('d r b\nc s a\n')
    checkpoint = save_checkpoint(make_checkpoint(initial_model('PCUltra-S', 42).state_dict(), variant='PCUltra-S', seed=42, updates=0,
                                                 weights='raw', order_sha256=VARIANTS['PCUltra-S'].pinned_order(42),
                                                 provenance={'source': 'test'}), tmp_path / 'S.pt')
    reports = {}
    for name, options in (('PCUltra', {'graph_inference_backend': 'torch'}), ('ULTRA', OPTIONS)):   # 'torch': the same operations
        args = Namespace()
        args.model, args.ultra_checkpoint = name, str(checkpoint)
        for key, value in options.items():
            setattr(args, key, value)
        args.dataset_dir, args.path_to_store_single_run = str(dataset), str(tmp_path / name)
        args.num_epochs, args.batch_size, args.separator = 0, 2, ' '
        args.scoring_technique, args.eval_model, args.eval_tie_policy = 'NegSample', 'test', 'pessimistic'
        reports[name] = Execute(args).start()
    assert reports['PCUltra']['Test'] == reports['ULTRA']['Test']
    assert (tmp_path / 'PCUltra' / 'pcultra_graph.pt').is_file()
    restored, reference = KGE(path=str(tmp_path / 'PCUltra')), KGE(path=str(tmp_path / 'ULTRA'))
    assert isinstance(restored.model, PCUltra) and (restored.model.relation_weighting, restored.model.pair_channels) == ('normalized', 'on')
    triples = torch.tensor([[0, 0, 1], [3, 1, 0]])
    assert torch.equal(restored.model(triples), reference.model(triples))
    restored.model.set_inference_backend('auto')   # the sparse product
    torch.testing.assert_close(restored.model(triples), reference.model(triples), rtol=1e-5, atol=1e-5)
    restored.predict_missing_tail_entity('a', 'r')
