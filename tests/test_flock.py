"""Flock checkpoint, fixed-walk parity, sampler, and DICE integration tests."""
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from dicee.config import Namespace
from dicee.models import Flock, FlockRelation
from dicee.models.flock_walks import WalkGraph, anonymize

FIXTURES = Path(__file__).parent / "fixtures" / "flock"


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def facts():
    return torch.tensor([[0, 0, 1], [0, 0, 2], [1, 1, 2], [2, 0, 3], [3, 1, 0]])


def make_model(cls, facts, **kwargs):
    args = dict(num_entities=5, num_relations=2, flock_dim=8, flock_walk_num=2,
                flock_walk_len=8, flock_refinements=2, flock_seed=42)
    args.update(kwargs)
    return cls(args).set_graph(facts)


@pytest.mark.parametrize("cls,keys,parameters", [(Flock, 206, 801969), (FlockRelation, 210, 810289)])
def test_official_architecture(cls, keys, parameters):
    model = cls({})
    assert len(model.state_dict()) == keys
    assert sum(p.numel() for p in model.parameters()) == parameters
    assert not hasattr(model, "entity_embeddings")
    with pytest.raises(RuntimeError, match="Attach"):
        model(torch.tensor([[0, 0, 1]]))


@pytest.mark.parametrize("task", ["entity", "relation"])
@pytest.mark.parametrize("tiny", [True, False])
def test_upstream_numerical_parity(task, tiny):
    fixture = torch.load(FIXTURES / (task + ("_tiny" if tiny else "_pretrained") + ".pt"), weights_only=True)
    cls = Flock if task == "entity" else FlockRelation
    model = cls(dict(num_entities=7, num_relations=3, flock_dim=fixture["dim"], flock_walk_num=fixture["walk_num"],
                     flock_walk_len=fixture["walk_len"], flock_refinements=fixture["refinements"], flock_query_batch_size=8))
    if tiny:
        model.load_state_dict(fixture["state_dict"], strict=True)
    else:
        directory = os.environ.get("FLOCK_CHECKPOINT_DIR")
        if directory is None:
            pytest.skip("Set FLOCK_CHECKPOINT_DIR to the directory containing both official Flock checkpoints")
        path = Path(directory) / ("flock_" + task + ".pth")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture["checkpoint_sha256"]
        model.load_pretrained(path)
    model.set_graph(fixture["triples"])
    for name, case in fixture["cases"].items():
        model.train(name == "training")

        def replay(graph, heads, tails, generator):
            actual = torch.cat((graph.edge_index.T, graph.edge_type[:, None]), 1)
            expected = torch.cat((case["edge_index"].T, case["edge_type"][:, None]), 1)
            assert set(map(tuple, actual.tolist())) == set(map(tuple, expected.tolist()))
            return case["records"]

        model._draw_walks = replay
        scores = model(case["groups"])
        torch.testing.assert_close(scores, case["scores"], atol=1e-5, rtol=1e-4)
        if name == "training":
            scores.sum().backward()
            for key, parameter in model.named_parameters():
                expected = fixture["gradients"][key]
                if expected is None:
                    assert parameter.grad is None, key
                else:
                    torch.testing.assert_close(parameter.grad, expected, atol=1e-5, rtol=1e-4, msg=lambda msg: key + ": " + msg)


def test_anonymization_against_compiled_reference():
    for task in ("entity", "relation"):
        fixture = torch.load(FIXTURES / (task + "_pretrained.pt"), weights_only=True)
        for case in fixture["cases"].values():
            nodes, node_names, _, _, types, type_names, _ = case["records"]
            torch.testing.assert_close(anonymize(nodes.flatten(0, 2)).reshape_as(nodes), node_names)
            torch.testing.assert_close(anonymize(types.flatten(0, 2), 6, 129).reshape_as(types), type_names)


def test_edge_parser_distribution_against_compiled_reference():
    fixture = torch.load(FIXTURES / "parser.pt", weights_only=True)
    facts = fixture["triples"]
    edges = torch.cat((facts[:, [0, 2, 1]], facts[:, [2, 0, 1]] + torch.tensor([0, 0, 3])))
    graph = WalkGraph(edges[:, :2].T, edges[:, 2], 7, 6)
    types, directions = graph.parse_types(fixture["walk"].expand(10000, -1), torch.Generator().manual_seed(42))
    counts = torch.stack([torch.bincount(types[:, i] * 4 + directions[:, i], minlength=28) for i in range(types.shape[1])])
    # Independent RNGs sample the same multinomial distributions, not the same draws.
    torch.testing.assert_close(counts / 10000, fixture["counts"] / 10000, atol=0.025, rtol=0)
    assert torch.equal(counts == 0, fixture["counts"] == 0)
    assert (directions[:, 3] == 1).all()  # official bool-queue self-loop convention
    assert (types[:, -2:] == 6).all() and (directions[:, -2:] == 3).all()


def test_transition_distributions_against_compiled_reference():
    fixture = torch.load(FIXTURES / "transitions.pt", weights_only=True)
    facts = fixture["triples"]
    edges = torch.cat((facts[:, [0, 2, 1]], facts[:, [2, 0, 1]] + torch.tensor([0, 0, 3])))
    graph = WalkGraph(edges[:, :2].T, edges[:, 2], 7, 6)
    records = graph.walk(torch.arange(7).repeat(2000), 6, True, torch.Generator().manual_seed(991))
    walks = records[0]
    first = torch.bincount(walks[:, 0] * 7 + walks[:, 1], minlength=49).view(7, 7)
    second = torch.bincount((walks[:, 0] * 7 + walks[:, 1]) * 7 + walks[:, 2], minlength=343).view(49, 7)
    for actual, expected in ((first, fixture["first"]), (second, fixture["second"])):
        assert torch.equal(actual == 0, expected == 0)
        torch.testing.assert_close(actual / actual.sum(1, keepdim=True).clamp_min(1),
                                   expected / expected.sum(1, keepdim=True).clamp_min(1), atol=0.07, rtol=0)


@pytest.mark.parametrize("cls", [Flock, FlockRelation])
def test_sampler_prefixes_and_non_backtracking(cls, facts):
    model = make_model(cls, facts)
    heads, tails = torch.tensor([0, 4]), torch.tensor([3, 1])
    records = model.sample_walks(heads, tails if cls is FlockRelation else None)
    walks, names, restarts, neighbors, types, type_names, directions = records
    assert walks.shape == (2, 2, 8 if cls is FlockRelation else 6, 8)
    assert torch.equal(walks[:, :, :2, 0], heads[None, :, None].expand(2, 2, 2))
    if cls is FlockRelation:
        assert torch.equal(walks[:, :, 2:4, 0], tails[None, :, None].expand(2, 2, 2))
    assert not restarts.any() and not neighbors.any()
    assert (names[..., 0] == 1).all()
    assert (types[..., 0] == 4).all() and (type_names[..., 0] == 9).all()
    assert (directions[..., 0] == 3).all()
    assert (directions[:, :, -2:, 1] == 0).all()
    graph_triples = set(map(tuple, model.graph_triples.tolist()))
    for node_walk, type_walk, direction_walk in zip(walks.flatten(0, 2), types.flatten(0, 2), directions.flatten(0, 2)):
        for step in range(1, len(node_walk)):
            previous, node, relation, direction = map(int, (node_walk[step - 1], node_walk[step], type_walk[step], direction_walk[step]))
            if relation == 4:
                assert previous == node == 4
            else:
                triple = (previous, relation, node) if direction == 0 else (node, relation, previous)
                assert triple in graph_triples
            if step >= 2:
                options = {t for h, _, t in graph_triples if h == previous and t != previous}
                if len(options) > 1:
                    assert node != int(node_walk[step - 2])


@pytest.mark.parametrize("cls", [Flock, FlockRelation])
def test_replay_scoring_interfaces(cls, facts):
    model = make_model(cls, facts).eval()
    triple = facts[:1]
    query = triple[:, :2] if cls is Flock else triple[:, [0, 2]]
    target = triple[:, 2:] if cls is Flock else triple[:, 1:2]
    scores = model(query)
    torch.testing.assert_close(model(triple), scores.gather(1, target).flatten())
    torch.testing.assert_close(model((query, target)), scores.gather(1, target))
    torch.testing.assert_close(model(query, target), scores.gather(1, target))
    torch.testing.assert_close(model(query, torch.tensor([0, 1])), scores[:, :2])
    groups = triple[:, None].repeat(1, scores.shape[1], 1)
    groups[..., 2 if cls is Flock else 1] = torch.arange(scores.shape[1])
    torch.testing.assert_close(model(groups), scores)
    assert model(torch.empty(0, 3, dtype=torch.long)).shape == (0,)
    assert model(torch.empty(0, 2, dtype=torch.long)).shape == (0, scores.shape[1])
    assert model(query, torch.empty(0, dtype=torch.long)).shape == (1, 0)


@pytest.mark.parametrize("cls", [Flock, FlockRelation])
def test_stochastic_sampling_and_logit_ensemble(cls, facts):
    model = make_model(cls, facts, flock_seed=None).eval()
    query = facts[:1, :2] if cls is Flock else facts[:1, [0, 2]]
    torch.manual_seed(4321)
    expected = torch.stack([model(query) for _ in range(3)]).mean(0)
    model.test_samples = 3
    torch.manual_seed(4321)
    torch.testing.assert_close(model(query), expected)
    assert not torch.allclose(model(query), expected)
    model.seed = 123
    torch.testing.assert_close(model(query), model(query))
    model.train()
    # Test-time averaging does not multiply the training computation.
    first = model(query)
    model.test_samples = 1
    torch.testing.assert_close(model(query), first)


@pytest.mark.parametrize("cls", [Flock, FlockRelation])
def test_fixed_walk_equivariance(cls, facts):
    model = make_model(cls, facts).eval()
    heads = torch.tensor([0])
    query = torch.tensor([1])  # relation 1 or tail entity 1
    candidates = torch.arange(5 if cls is Flock else 2)[None]
    records = model.sample_walks(heads, query if cls is FlockRelation else None)
    before = model.score_walks(heads, query, candidates, records)
    entities, relations = torch.tensor([3, 4, 1, 0, 2]), torch.tensor([1, 0])
    renamed = torch.stack((entities[facts[:, 0]], relations[facts[:, 1]], entities[facts[:, 2]]), 1)
    model.set_graph(renamed)
    type_map = torch.cat((relations, relations + 2, torch.tensor([4])))
    mapped = list(records)
    mapped[0], mapped[4] = entities[records[0]], type_map[records[4]]
    new_query = relations[query] if cls is Flock else entities[query]
    new_candidates = entities[candidates] if cls is Flock else relations[candidates]
    actual = model.score_walks(entities[heads], new_query, new_candidates, tuple(mapped))
    torch.testing.assert_close(actual, before, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("cls", [Flock, FlockRelation])
def test_graph_lifecycle_and_checkpoint_validation(cls, facts, tmp_path):
    model = make_model(cls, facts).eval()
    before = model(facts[:1])
    model.set_graph(torch.cat((facts.flip(0), facts)))
    torch.testing.assert_close(model(facts[:1]), before)
    path = tmp_path / "graph.pt"
    model.save_graph(path)
    restored = make_model(cls, facts).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.load_graph(path)
    torch.testing.assert_close(restored(facts[:1]), before)
    assert not any("graph" in key or "edge_index" in key for key in model.state_dict())
    restored.set_graph(torch.tensor([[7, 2, 6]]), 8, 3)
    assert torch.isfinite(restored(torch.tensor([[7, 2, 5]]))).all()
    with pytest.raises(ValueError, match="vocabulary"):
        restored.load_graph(path)
    with pytest.raises(FileNotFoundError, match="artifact"):
        model.load_graph(tmp_path / "missing.pt")
    path = tmp_path / "checkpoint.pth"
    for state in (model.state_dict(), {"model": model.state_dict()}):
        torch.save(state, path)
        model.load_pretrained(path)
    for state in (Flock({}).state_dict(), {**model.state_dict(), "extra": torch.ones(1)}, {}):
        torch.save(state, path)
        with pytest.raises(ValueError, match="architecture mismatch"):
            model.load_pretrained(path)


def test_entity_inverse_conditioning(facts):
    model = make_model(Flock, facts).eval()
    before = model.forward_k_vs_all_heads(facts[:1, 1:])
    inverse = facts[:, [2, 1, 0]].clone()
    forward = facts.clone()
    forward[:, 1] *= 2
    inverse[:, 1] = inverse[:, 1] * 2 + 1
    model.set_graph(torch.cat((forward, inverse)), 5, 4, {0: 1, 2: 3})
    torch.testing.assert_close(model(inverse[:1, :2]), before)
    torch.testing.assert_close(model(inverse[:1]), before[:, 0])
    torch.testing.assert_close(model.forward_k_vs_all_heads(forward[:1, 1:], torch.tensor([0])), before[:, :1])


@pytest.mark.parametrize("cls", [Flock, FlockRelation])
def test_training_masks_and_empty_graph(cls, facts):
    model = make_model(cls, facts)
    captured = []
    draw = model._draw_walks

    def capture(graph, *args):
        captured.append(set(map(tuple, torch.stack((graph.edge_index[0], graph.edge_type, graph.edge_index[1]), 1).tolist())))
        return draw(graph, *args)

    model._draw_walks = capture
    query = facts[:1, :2] if cls is Flock else facts[:1, [0, 2]]
    scores = model(query)
    assert (0, 0, 1) not in captured[0] and (1, 2, 0) not in captured[0]
    if cls is Flock:
        assert (0, 0, 2) not in captured[0] and (2, 2, 0) not in captured[0]
    scores.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.set_graph(facts[:1])
    scores = model(facts[:1])  # all message-passing facts are masked
    assert captured[-1] == set()
    assert torch.isfinite(scores).all()
    scores.sum().backward()


@pytest.mark.parametrize("cls", [Flock, FlockRelation])
def test_invalid_settings_and_ids(cls, facts):
    for args in ({"flock_dim": 0}, {"flock_query_batch_size": 0}, {"flock_refinements": 0}, {"flock_walk_len": 1},
                 {"flock_walk_num": 0}, {"flock_test_samples": 0}, {"flock_attention_heads": 3}, {"flock_seed": -1},
                 {"normalization": "LayerNorm"}, {"byte_pair_encoding": True}):
        with pytest.raises(ValueError):
            cls(args)
    model = make_model(cls, facts).eval()
    for triples in (torch.tensor([[5, 0, 1]]), torch.tensor([[0, 2, 1]]), torch.tensor([[-1, 0, 1]])):
        with pytest.raises(ValueError, match="vocabulary"):
            model(triples)
    with pytest.raises(ValueError, match="Inverse"):
        model.set_graph(facts, inverse_relations={0: 0})
    with pytest.raises(ValueError):
        model(torch.tensor([[0, 1]]), torch.tensor([-1]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("cls", [Flock, FlockRelation])
def test_cuda_graph_movement_and_replay(cls, facts):
    model = make_model(cls, facts).eval()
    before = model(facts[:1])
    model.cuda()
    assert model.graph_triples.is_cuda
    # Walker context intentionally stays on CPU; model records/states use CUDA.
    torch.testing.assert_close(model(facts[:1]).cpu(), before, atol=1e-5, rtol=1e-4)
    model.train()
    model(facts[:1]).sum().backward()
    assert all(p.grad.is_cuda for p in model.parameters() if p.grad is not None)
    model.set_graph(facts)
    assert model.graph_triples.is_cuda


@pytest.mark.parametrize("name,technique,grouped,trainer,backend", [
    ("Flock", "NegSample", False, "torchCPUTrainer", "pandas"),
    ("Flock", "NegSample", True, "torchCPUTrainer", "pandas"),
    ("Flock", "NegSample", True, "PL", "pandas"),
    ("Flock", "KvsAll", False, "torchCPUTrainer", "polars"),
    ("Flock", "KvsSample", False, "torchCPUTrainer", "pandas"),
    ("Flock", "1vsSample", False, "torchCPUTrainer", "pandas"),
    ("Flock", "1vsAll", False, "torchCPUTrainer", "pandas"),
    ("Flock", "FixedNegSample", False, "torchCPUTrainer", "pandas"),
    ("FlockRelation", "KvsAll", False, "torchCPUTrainer", "pandas"),
    ("FlockRelation", "KvsAll", False, "PL", "polars"),
])
def test_execute_and_reload(tmp_path, name, technique, grouped, trainer, backend):
    from dicee.executer import Execute
    from dicee.knowledge_graph_embeddings import KGE
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "train.txt").write_text("a r b\na r c\nb s c\nc r d\nd s a\n")
    (dataset / "valid.txt").write_text("b r d\n")
    (dataset / "test.txt").write_text("d r b\n")
    args = Namespace()
    args.model, args.flock_dim = name, 8
    args.flock_walk_num, args.flock_walk_len, args.flock_refinements, args.flock_seed = 1, 6, 2, 123
    args.dataset_dir, args.path_to_store_single_run = str(dataset), str(tmp_path / "run")
    args.num_epochs, args.batch_size, args.neg_ratio, args.lr = 1, 2, 2, 0.001
    args.scoring_technique, args.trainer = technique, trainer
    args.backend, args.separator = backend, " "
    args.strict_negative_sampling = grouped
    args.adversarial_temperature = 1.0 if grouped else None
    if trainer == "PL":
        args.pl_trainer_kwargs = {"accelerator": "cpu", "devices": 1}
    execute = Execute(args)
    report = execute.start()
    assert np.isfinite(report["Test"]["MRR"])
    restored = KGE(path=str(tmp_path / "run"))
    triples = torch.tensor([[0, 0, 1]])
    torch.testing.assert_close(restored.model(triples), execute.trained_model(triples))
    assert (tmp_path / "run" / "flock_graph.pt").is_file()
    if name == "Flock":
        restored.predict_missing_head_entity("r", "b")
        restored.predict_missing_tail_entity("a", "r")
    else:
        assert len(restored.relation_to_idx) == 2
        scores = restored.predict_missing_relations("a", "b")
        pairs = torch.tensor([[restored.entity_to_idx["a"], restored.entity_to_idx["b"]]])
        torch.testing.assert_close(scores, restored.model(pairs).flatten())
    ensemble = KGE(path=str(tmp_path / "run"), construct_ensemble=True)
    torch.testing.assert_close(ensemble.model(triples), restored.model(triples))


@pytest.mark.parametrize("name", ["Flock", "FlockRelation"])
def test_pipeline_rejects_unsupported_settings(name, tmp_path):
    from dicee.static_preprocess_funcs import preprocesses_input_args
    for overrides in ({"trainer": "torchFSDP"}, {"byte_pair_encoding": True}, {"save_embeddings_as_csv": True},
                      {"num_folds_for_cv": 2}, {"pl_trainer_kwargs": {"devices": 2}}, {"normalization": "LayerNorm"}):
        args = Namespace()
        args.model, args.dataset_dir, args.scoring_technique = name, str(tmp_path), "KvsAll"
        for key, value in overrides.items():
            setattr(args, key, value)
        with pytest.raises(ValueError):
            preprocesses_input_args(args)
