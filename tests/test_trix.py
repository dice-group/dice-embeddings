"""Official TRIX parity and DICE graph/scoring/training integration contracts."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from dicee.config import Namespace
from dicee.models.trix import INTERACTIONS, TRIX, TRIXRelation

FIXTURES = Path(__file__).parent / "fixtures" / "trix"


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
    return cls(dict(num_entities=5, num_relations=2, trix_dim=8, **kwargs)).set_graph(facts)


@pytest.mark.parametrize("cls,keys,parameters", [(TRIX, 154, 87138), (TRIXRelation, 256, 147489)])
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
    fixture = torch.load(FIXTURES / (task + ("_tiny" if tiny else "_prediction") + ".pt"), weights_only=True)
    cls = TRIX if task == "entity" else TRIXRelation
    model = cls(dict(num_entities=fixture["num_entities"], num_relations=fixture["num_relations"], trix_dim=fixture["dim"]))
    if tiny:
        model.load_state_dict(fixture["state_dict"], strict=True)
    else:
        directory = os.environ.get("TRIX_CHECKPOINT_DIR")
        if directory is None:
            pytest.skip("Set TRIX_CHECKPOINT_DIR to the official checkout with both checkpoints")
        path = Path(directory) / (task + "_prediction.pth")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture["checkpoint_sha256"]
        model.load_pretrained(path)
    model.set_graph(fixture["triples"]).eval()
    for role, (index, types) in model.relation_graph.items():
        expected_index, expected_type = fixture["relation_graph"][role]
        actual = torch.cat((index.T, types[:, None]), 1)
        expected = torch.cat((expected_index.T, expected_type[:, None]), 1)
        assert set(map(tuple, actual.tolist())) == set(map(tuple, expected.tolist()))
    queries = fixture["queries"]
    if task == "entity":
        torch.testing.assert_close(model._reason(queries[:, 0], queries[:, 1], queries[:, 1], (model.edge_index, model.edge_type)),
                                   fixture["relations"], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(model(queries[:, :2]), fixture["tails"], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(model.forward_k_vs_all_heads(queries[:, 1:]), fixture["heads"], atol=1e-5, rtol=1e-4)
    else:
        torch.testing.assert_close(model(queries[:, [0, 2]]), fixture["scores"], atol=1e-5, rtol=1e-4)
    model.train()
    scores = model(fixture["grouped"])
    torch.testing.assert_close(scores, fixture["training_scores"], atol=1e-5, rtol=1e-4)
    scores.sum().backward()
    for name, parameter in model.named_parameters():
        expected = fixture["gradients"][name]
        if expected is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, expected, atol=1e-5, rtol=1e-4, msg=lambda msg: name + ": " + msg)


@pytest.mark.parametrize("cls", [TRIX, TRIXRelation])
def test_scoring_interfaces_and_chunking(cls, facts):
    model = make_model(cls, facts).eval()
    queries = facts[:, :2] if cls is TRIX else facts[:, [0, 2]]
    target = facts[:, 2:] if cls is TRIX else facts[:, 1:2]
    all_scores = model(queries)
    torch.testing.assert_close(model(facts), all_scores.gather(1, target).flatten())
    torch.testing.assert_close(model((queries, target)), all_scores.gather(1, target))
    torch.testing.assert_close(model(queries, target), all_scores.gather(1, target))
    torch.testing.assert_close(model(queries, torch.tensor([0, 1])), all_scores[:, :2])
    groups = facts[:, None].repeat(1, all_scores.shape[1], 1)
    groups[..., 2 if cls is TRIX else 1] = torch.arange(all_scores.shape[1])
    torch.testing.assert_close(model(groups), all_scores)
    torch.testing.assert_close(model(facts[[2, 1, 2, 0]]), model(facts)[[2, 1, 2, 0]])
    model.query_batch_size = 1
    torch.testing.assert_close(model(queries), all_scores)
    assert torch.isfinite(all_scores).all()
    assert model(torch.empty(0, 3, dtype=torch.long)).shape == (0,)
    assert model(torch.empty(0, 2, dtype=torch.long)).shape == (0, all_scores.shape[1])


@pytest.mark.parametrize("cls", [TRIX, TRIXRelation])
def test_graph_lifecycle_and_equivariance(cls, facts, tmp_path):
    model = make_model(cls, facts).eval()
    before = model(facts)
    model.set_graph(torch.cat((facts.flip(0), facts)))
    torch.testing.assert_close(model(facts), before)
    entities, relations = torch.tensor([3, 4, 1, 0, 2]), torch.tensor([1, 0])
    permuted = torch.stack((entities[facts[:, 0]], relations[facts[:, 1]], entities[facts[:, 2]]), 1)
    model.set_graph(permuted)
    torch.testing.assert_close(model(permuted), before, atol=1e-5, rtol=1e-4)
    model.set_graph(facts)
    model.save_graph(tmp_path / "trix_graph.pt")
    restored = make_model(cls, facts).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.load_graph(tmp_path / "trix_graph.pt")
    torch.testing.assert_close(restored(facts), before)
    assert not any("edge_index" in key or "graph" in key or "relation_id_map" in key for key in model.state_dict())
    model.set_graph(torch.tensor([[4, 0, 0], [0, 1, 2]]))
    assert model.edge_type.numel() == 4
    assert not torch.allclose(model(facts), before)
    # Transfer learned weights to a different vocabulary without resizing them.
    model.set_graph(torch.tensor([[7, 2, 6]]), num_entities=8, num_relations=3)
    assert torch.isfinite(model(torch.tensor([[7, 2, 5]]))).all()
    with pytest.raises(ValueError, match="vocabulary"):
        model.load_graph(tmp_path / "trix_graph.pt")
    with pytest.raises(FileNotFoundError, match="artifact"):
        model.load_graph(tmp_path / "missing.pt")


@pytest.mark.parametrize("cls", [TRIX, TRIXRelation])
def test_inverse_mapping(cls, facts):
    model = make_model(cls, facts).eval()
    before = model(facts)
    heads = model.forward_k_vs_all_heads(facts[:, 1:])
    inverse = facts[:, [2, 1, 0]].clone()
    # Interleave direct/inverse IDs in the external vocabulary.
    forward = facts.clone()
    forward[:, 1] *= 2
    inverse[:, 1] = inverse[:, 1] * 2 + 1
    model.set_graph(torch.cat((forward, inverse)), 5, 4, {0: 1, 2: 3})
    torch.testing.assert_close(model(forward), before)
    if cls is TRIX:
        torch.testing.assert_close(model(inverse[:, :2]), heads)
        torch.testing.assert_close(model(inverse), heads.gather(1, facts[:, :1]).flatten())
    else:
        torch.testing.assert_close(model(forward[:, [0, 2]])[:, ::2].gather(1, facts[:, 1:2]).flatten(), before)


@pytest.mark.parametrize("cls", [TRIX, TRIXRelation])
def test_training_masks_and_gradients(cls, facts):
    model = make_model(cls, facts)
    graph = model.graph_triples.clone()
    targets = model._convert(facts[:1])
    index, types = model._training_edges(targets=targets)
    kept = set(map(tuple, torch.stack((index[0], types, index[1]), 1).tolist()))
    assert (0, 0, 1) not in kept and (1, 2, 0) not in kept
    assert (0, 0, 2) in kept
    captured = []
    entity = model.entity_model_1 if cls is TRIX else model.entity_model[0]
    hook = entity.layers[0].register_forward_pre_hook(lambda module, args: captured.append((args[2], args[3])))
    queries = facts[:1, :2] if cls is TRIX else facts[:1, [0, 2]]
    scores = model(queries)
    hook.remove()
    index, types = captured[0]
    kept = set(map(tuple, torch.stack((index[0], types, index[1]), 1).tolist()))
    assert (0, 0, 1) not in kept and (1, 2, 0) not in kept
    if cls is TRIX:
        assert (0, 0, 2) not in kept and (2, 2, 0) not in kept
    scores.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    torch.testing.assert_close(model.graph_triples, graph)


@pytest.mark.parametrize("cls", [TRIX, TRIXRelation])
def test_checkpoint_validation(cls, facts, tmp_path):
    model = make_model(cls, facts)
    state = model.state_dict()
    path = tmp_path / "checkpoint.pth"
    for wrapped in (False, True):
        torch.save({"model": state} if wrapped else state, path)
        model.load_pretrained(path)
    for invalid in ({**state, "extra": torch.ones(1)}, dict(list(state.items())[1:]), TRIX({}).state_dict()):
        torch.save({"model": invalid}, path)
        with pytest.raises(ValueError, match="architecture mismatch"):
            model.load_pretrained(path)


@pytest.mark.parametrize("cls", [TRIX, TRIXRelation])
def test_invalid_settings_and_ids(cls, facts):
    for args in ({"trix_dim": 0}, {"trix_query_batch_size": 0}, {"normalization": "LayerNorm"}, {"byte_pair_encoding": True}):
        with pytest.raises(ValueError):
            cls(args)
    model = make_model(cls, facts).eval()
    for triples in (torch.tensor([[5, 0, 1]]), torch.tensor([[0, 2, 1]]), torch.tensor([[-1, 0, 1]])):
        with pytest.raises(ValueError, match="vocabulary"):
            model(triples)
    for inverse in ({0: 0}, {0: 3}, {0: 1, 1: 0}):
        with pytest.raises(ValueError, match="Inverse"):
            model.set_graph(facts, inverse_relations=inverse)
    for candidates in (torch.tensor([-1]), torch.tensor([10]), torch.zeros(2, 2, dtype=torch.long)):
        with pytest.raises(ValueError):
            model(torch.tensor([[0, 1]]), candidates)


@pytest.mark.parametrize("cls", [TRIX, TRIXRelation])
def test_empty_relation_roles_and_self_loops(cls):
    model = cls(dict(num_entities=2, num_relations=1, trix_dim=8)).set_graph(torch.tensor([[0, 0, 1]])).eval()
    assert model.relation_graph["hh"][0].numel() == model.relation_graph["tt"][0].numel() == 0
    assert torch.isfinite(model(torch.tensor([[0, 0, 1], [1, 0, 1]]))).all()
    model.set_graph(torch.tensor([[0, 0, 0]]))
    assert torch.isfinite(model(torch.tensor([[0, 0, 0]]))).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("cls", [TRIX, TRIXRelation])
def test_cuda_graph_movement(cls, facts):
    model = make_model(cls, facts).eval()
    before = model(facts)
    model.cuda()
    assert model.graph_triples.is_cuda
    assert all(tensor.is_cuda for edges in model.relation_graph.values() for tensor in edges)
    torch.testing.assert_close(model(facts).cpu(), before, atol=1e-5, rtol=1e-4)
    model.train()
    model(facts).sum().backward()
    assert all(p.grad.is_cuda for p in model.parameters() if p.grad is not None)
    model.set_graph(facts)
    assert all(getattr(model, "rel_edge_index_" + role).is_cuda for role in INTERACTIONS)


@pytest.mark.parametrize("model_name,technique,grouped,trainer,backend", [
    ("TRIX", "NegSample", False, "torchCPUTrainer", "pandas"),
    ("TRIX", "NegSample", True, "torchCPUTrainer", "pandas"),
    ("TRIX", "NegSample", True, "PL", "pandas"),
    ("TRIX", "KvsAll", False, "torchCPUTrainer", "polars"),
    ("TRIX", "KvsSample", False, "torchCPUTrainer", "pandas"),
    ("TRIX", "1vsSample", False, "torchCPUTrainer", "pandas"),
    ("TRIX", "1vsAll", False, "torchCPUTrainer", "pandas"),
    ("TRIX", "FixedNegSample", False, "torchCPUTrainer", "pandas"),
    ("TRIXRelation", "KvsAll", False, "torchCPUTrainer", "pandas"),
    ("TRIXRelation", "KvsAll", False, "PL", "polars"),
])
def test_execute_and_reload(tmp_path, model_name, technique, grouped, trainer, backend):
    from dicee.executer import Execute
    from dicee.knowledge_graph_embeddings import KGE
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "train.txt").write_text("a r b\na r c\nb s c\nc r d\nd s a\n")
    (dataset / "valid.txt").write_text("b r d\n")
    (dataset / "test.txt").write_text("d r b\n")
    args = Namespace()
    args.model, args.trix_dim = model_name, 8
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
    assert (tmp_path / "run" / "trix_graph.pt").is_file()
    if model_name == "TRIX":
        restored.predict_missing_head_entity("r", "b")
        restored.predict_missing_tail_entity("a", "r")
    else:
        assert len(restored.relation_to_idx) == 2
        scores = restored.predict_missing_relations("a", "b")
        expected = restored.model(torch.tensor([[restored.entity_to_idx["a"], restored.entity_to_idx["b"]]]))
        torch.testing.assert_close(scores, expected.flatten())
    ensemble = KGE(path=str(tmp_path / "run"), construct_ensemble=True)
    torch.testing.assert_close(ensemble.model(triples), restored.model(triples))


@pytest.mark.parametrize("cls", [TRIX, TRIXRelation])
def test_zero_epoch_checkpoint(cls, tmp_path):
    from dicee.executer import Execute
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "train.txt").write_text("a r b\nb s c\nc r a\n")
    (dataset / "test.txt").write_text("a s c\n")
    path = tmp_path / "checkpoint.pth"
    pretrained = cls({})
    torch.save({"model": pretrained.state_dict()}, path)
    args = Namespace()
    args.model, args.trix_checkpoint = cls.name, str(path)
    args.dataset_dir, args.path_to_store_single_run = str(dataset), str(tmp_path / "run")
    args.num_epochs, args.batch_size = 0, 2
    args.scoring_technique, args.eval_model = "KvsAll", "test"
    report = Execute(args).start()
    state = torch.load(tmp_path / "run" / "model.pt", weights_only=True)
    for key in state:
        torch.testing.assert_close(state[key], pretrained.state_dict()[key])
    assert np.isfinite(report["Test"]["MRR"])
    assert json.loads((tmp_path / "run" / "configuration.json").read_text())["trix_dim"] == 32
