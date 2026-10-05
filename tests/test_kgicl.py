"""Official KG-ICL parity and DICE graph, prompt, scoring and training contracts."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from dicee.config import Namespace
from dicee.models.kgicl import KGICL, UPSTREAM_COMMIT
from dicee.models.kgicl_prompts import PromptGraph, PromptSampler, answer_distance_rates

FIXTURES = Path(__file__).parent / "fixtures" / "kgicl"
SMALL = dict(kgicl_dim=8, kgicl_attn_dim=3, kgicl_num_layers=3, kgicl_prompt_layers=2)


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)
    torch.set_float32_matmul_precision(previous_precision)


@pytest.fixture
def facts():
    return torch.tensor([[0, 0, 1], [0, 0, 2], [1, 1, 2], [2, 0, 3], [3, 1, 0], [3, 2, 4], [4, 0, 0],
                         [1, 0, 3], [2, 2, 2], [4, 1, 1], [0, 2, 3], [2, 0, 4]])


def make_model(facts, **kwargs):
    return KGICL(dict(num_entities=6, num_relations=3, **SMALL, **kwargs)).set_graph(facts)


def load_fixture(name):
    fixture = torch.load(FIXTURES / f"{name}.pt", weights_only=False)
    assert fixture["upstream_commit"] == UPSTREAM_COMMIT
    assert fixture["patch_sha256"] == hashlib.sha256((FIXTURES / "upstream-fixes.patch").read_bytes()).hexdigest()
    args = dict(num_entities=fixture["num_entities"], num_relations=fixture["num_relations"], kgicl_dim=fixture["dim"],
                kgicl_attn_dim=fixture["attn_dim"], kgicl_num_layers=fixture["num_layers"],
                kgicl_prompt_layers=fixture["prompt_layers"], kgicl_shots=fixture["shots"])
    model = KGICL(args)
    if name == "tiny":
        state = model.state_dict()
        assert set(fixture["state_dict"]) == {k for k in state if ".conv." not in k}
        model.load_state_dict({**state, **fixture["state_dict"]}, strict=True)
    else:
        directory = os.environ.get("KGICL_CHECKPOINT_DIR")
        if directory is None:
            pytest.skip("Set KGICL_CHECKPOINT_DIR to the directory with the official KG-ICL-6L model_best.tar")
        path = Path(directory) / "model_best.tar"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture["checkpoint_sha256"]
        model.load_pretrained(path)
    return fixture, model.set_graph(fixture["graph"])


def replay(fixture):
    return {q: [PromptGraph(g["edge_index"], g["edge_type"], g["labels"], g["head"], g["tail"]) for g in graphs]
            for q, graphs in fixture["prompts"].items()}


def score_queries(model, queries, num_direct):
    """All-entity scores for internal (head, relation) queries; inverse relations use head prediction."""
    rows = []
    for head, relation in queries.tolist():
        if relation < num_direct:
            rows.append(model(torch.tensor([[head, relation]])))
        else:
            rows.append(model.forward_k_vs_all_heads(torch.tensor([[relation - num_direct, head]])))
    return torch.cat(rows)


def test_official_architecture():
    model = KGICL({})
    assert len(model.state_dict()) == 220
    assert sum(p.numel() for p in model.parameters()) == 90856
    assert not hasattr(model, "entity_embeddings")
    with pytest.raises(RuntimeError, match="Attach"):
        model(torch.tensor([[0, 0, 1]]))


@pytest.mark.parametrize("name", ["tiny", "official"])
def test_upstream_numerical_parity(name):
    fixture, model = load_fixture(name)
    model.eval().use_prompts(replay(fixture))
    tolerance = dict(atol=2e-5, rtol=1e-4)
    with torch.no_grad():
        for relation, expected in fixture["prompt_outputs"].items():
            torch.testing.assert_close(model.encode_prompts(relation), expected, **tolerance)
        scores = score_queries(model, fixture["queries"], fixture["num_relations"])
        torch.testing.assert_close(scores, fixture["unmasked_scores"], **tolerance)
        rates = answer_distance_rates(fixture["valid"], fixture["train"], fixture["num_entities"])
        assert rates == fixture["answer_distance"]
        model.masked_distances = tuple(d for d in range(fixture["num_layers"] + 1) if rates[d] < 0.02)
        model.clear_inference_cache()
        masked = score_queries(model, fixture["queries"], fixture["num_relations"])
        torch.testing.assert_close(masked, fixture["scores"], **tolerance)
        assert (masked == 0).sum() > (scores == 0).sum()
    model.masked_distances = ()
    model.clear_inference_cache()
    # Evaluation-mode gradients; upstream's checkpoint keeps unused NBFNet parameters.
    (score_queries(model, fixture["queries"], fixture["num_relations"]) * fixture["gradient_weights"]).sum().backward()
    for parameter_name, parameter in model.named_parameters():
        expected = fixture["gradients"].get(parameter_name)
        if expected is None:
            assert parameter.grad is None, parameter_name
        else:
            torch.testing.assert_close(parameter.grad, expected, atol=1e-4, rtol=1e-4,
                                       msg=lambda msg: parameter_name + ": " + msg)


def test_prompt_extraction_matches_upstream():
    fixture = torch.load(FIXTURES / "tiny.pt", weights_only=False)
    sampler = PromptSampler(fixture["train"], fixture["num_entities"], fixture["num_relations"],
                            list(range(fixture["num_relations"])))
    checked = 0
    for relation, cases in fixture["cases"].items():
        for case in cases:
            example = sampler.extract(*case["example"])
            entities = example.entities.tolist()
            assert sorted(entities) == sorted(case["entities"])
            assert dict(zip(entities, example.labels.tolist())) == dict(zip(case["entities"], case["labels"]))
            triples = {(entities[h], r, entities[t]) for h, r, t in example.triples.tolist()}
            assert triples == set(case["triples"]) and len(triples) == len(example.triples)
            assert example.self_loop == (case["example"][0] == case["example"][1])
            checked += 1
    assert checked > 40 and any(c["example"][0] == c["example"][1] for cases in fixture["cases"].values() for c in cases)


def test_prompt_sampling_is_seeded_and_order_independent(facts):
    model = make_model(facts).eval()
    other = make_model(facts).eval()
    for relation in (0, 1, 2, 4, 5):
        first = model.prompt_graphs(relation)
        assert len(first) == 5
        again = other.prompt_graphs(relation)
        for a, b in zip(first, again):
            assert torch.equal(a.edge_index, b.edge_index) and torch.equal(a.labels, b.labels)
            assert (a.head, a.tail) == (b.head, b.tail)
    # Relation 0 has six facts: five distinct examples. Relation 1 has three: slots cycle.
    assert len(model.sampler.examples(0)) == 5 and len(model.sampler.examples(1)) == 3
    slots = model.prompt_graphs(1)
    assert slots[3] is not slots[0] and torch.equal(slots[3].edge_index, slots[0].edge_index)
    # Inverse relations reuse the examples with swapped head and tail tokens.
    assert [(g.head, g.tail) for g in model.prompt_graphs(4)] == [(1, 0) if not e.self_loop else (0, 0)
                                                                  for e in [model.sampler.examples(1)[j % 3] for j in range(5)]]
    seeded = make_model(facts, kgicl_prompt_seed=1).eval()
    assert any(not torch.equal(a.entities, b.entities) for a, b in zip(model.sampler.examples(0), seeded.sampler.examples(0)))
    queries = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 0], [4, 1], [0, 2]])
    with torch.no_grad():
        expected = model(queries)
        reversed_model = make_model(facts).eval()
        reversed_model.load_state_dict(model.state_dict())
        actual = reversed_model(queries.flip(0)).flip(0)
        reversed_model.query_batch_size = 1
        single = reversed_model(queries)
    assert torch.equal(expected, actual) and torch.equal(expected, single)


def test_self_loop_example_puts_both_tokens_on_the_example_entity():
    model = KGICL(dict(num_entities=4, num_relations=1, **SMALL)).set_graph(torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 2]]))
    examples = model.sampler.examples(0)
    loop = next(e for e in examples if e.self_loop)
    assert loop.labels[0].tolist() == [0, 0] and loop.entities[0] == 0
    graphs = model.sampler.prompts(0, [next(i for i, e in enumerate(examples) if e is loop)])
    assert (graphs[0].head, graphs[0].tail) == (0, 0)
    # Each self-loop fact appears once, with its inverse.
    assert (graphs[0].edge_index[0] == graphs[0].edge_index[1]).sum() == 2


def test_relation_without_facts_uses_upstream_fallback(facts):
    model = KGICL(dict(num_entities=6, num_relations=4, **SMALL)).set_graph(facts).eval()
    for relation, (head, tail) in ((3, (0, 1)), (7, (1, 0))):
        graphs = model.prompt_graphs(relation)
        assert len(graphs) == 5 and all(g is graphs[0] for g in graphs)
        assert graphs[0].edge_index.tolist() == [[0], [1]] and graphs[0].edge_type.tolist() == [relation]
        assert (graphs[0].head, graphs[0].tail) == (head, tail)
    with torch.no_grad():
        assert torch.isfinite(model(torch.tensor([[0, 3]]))).all()


@pytest.mark.parametrize("seed", [0, 3])
def test_scoring_interfaces_and_chunking(facts, seed):
    model = make_model(facts, kgicl_prompt_seed=seed).eval()
    queries, target = facts[:, :2], facts[:, 2:]
    with torch.no_grad():
        all_scores = model(queries)
        torch.testing.assert_close(model(facts), all_scores.gather(1, target).flatten())
        torch.testing.assert_close(model((queries, target)), all_scores.gather(1, target))
        torch.testing.assert_close(model(queries, torch.tensor([0, 1])), all_scores[:, :2])
        groups = facts[:, None].repeat(1, all_scores.shape[1], 1)
        groups[..., 2] = torch.arange(all_scores.shape[1])
        torch.testing.assert_close(model(groups), all_scores)
        torch.testing.assert_close(model(facts[[2, 1, 2, 0]]), model(facts)[[2, 1, 2, 0]])
        heads = model.forward_k_vs_all_heads(facts[:, 1:])
        assert torch.isfinite(all_scores).all() and torch.isfinite(heads).all()
        assert model(torch.empty(0, 3, dtype=torch.long)).shape == (0,)
        assert model(torch.empty(0, 2, dtype=torch.long)).shape == (0, all_scores.shape[1])
    # Entity 5 has no facts: it is never reached and scores exactly zero.
    assert (all_scores[:, 5] == 0).all()


def test_reached_set_and_distance_mask(facts):
    model = make_model(facts).eval()
    with torch.no_grad():
        scores = model(torch.tensor([[0, 0]]))
        _, depth = model._expand_scores(torch.tensor([0]), torch.tensor([0]),
                                        model.encode_prompts(0)[None], (model.edge_index, model.edge_type))
        model.masked_distances = (0, 1)
        model.clear_inference_cache()
        masked = model(torch.tensor([[0, 0]]))
    assert depth[0, 0] == 0 and depth[0, 5] == -1 and (depth[0, [1, 2, 3, 4]] == 1).all()
    assert (masked[0, :5] == 0).all() and scores[0, :5].ne(0).all()


def test_graph_lifecycle_and_equivariance(facts, tmp_path):
    model = make_model(facts).eval()
    prompts = {q: model.prompt_graphs(q) for q in range(6)}
    model.use_prompts(prompts)
    with torch.no_grad():
        before = model(facts)
    entities = torch.tensor([3, 5, 1, 0, 2, 4])
    permuted = torch.stack((entities[facts[:, 0]], facts[:, 1], entities[facts[:, 2]]), 1)
    model.set_graph(permuted).use_prompts(prompts)
    with torch.no_grad():
        torch.testing.assert_close(model(permuted), before, atol=1e-5, rtol=1e-4)
    model.set_graph(facts)
    assert not model._prompt_overrides
    model.save_graph(tmp_path / "kgicl_graph.pt")
    restored = make_model(facts).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.load_graph(tmp_path / "kgicl_graph.pt")
    with torch.no_grad():
        torch.testing.assert_close(restored(facts), model(facts))
    assert not any("edge_index" in key or "graph" in key or "relation_id_map" in key for key in model.state_dict())
    model.set_graph(torch.tensor([[7, 2, 6]]), num_entities=8, num_relations=3)
    with torch.no_grad():
        assert torch.isfinite(model(torch.tensor([[7, 2, 5]]))).all()
    with pytest.raises(ValueError, match="vocabulary"):
        model.load_graph(tmp_path / "kgicl_graph.pt")


def test_inverse_mapping(facts):
    model = make_model(facts).eval()
    prompts = {q: model.prompt_graphs(q) for q in range(6)}
    model.use_prompts(prompts)
    with torch.no_grad():
        before, heads = model(facts), model.forward_k_vs_all_heads(facts[:, 1:])
    forward, inverse = facts.clone(), facts[:, [2, 1, 0]].clone()
    forward[:, 1] *= 2
    inverse[:, 1] = inverse[:, 1] * 2 + 1
    model.set_graph(torch.cat((forward, inverse)), 6, 6, {0: 1, 2: 3, 4: 5}).use_prompts(prompts)
    with torch.no_grad():
        torch.testing.assert_close(model(forward), before)
        torch.testing.assert_close(model(inverse[:, :2]), heads)
    # Prompt examples are seeded by the public direct relation ID.
    assert model.sampler.public_relations == [0, 2, 4]


def test_training_masks_and_gradients(facts):
    model = make_model(facts)
    captured = []
    original = model._expand_scores
    model._expand_scores = lambda *a: captured.append(a[3]) or original(*a)
    scores = model(facts[:1, :2])
    index, types = captured[0]
    kept = set(map(tuple, torch.stack((index[0], types, index[1]), 1).tolist()))
    assert (0, 0, 1) not in kept and (1, 3, 0) not in kept and (0, 0, 2) not in kept
    assert (1, 1, 2) in kept
    scores.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.relation_encoder.W_message[0].weight.grad.abs().sum() > 0
    assert model.gnn_layers[0].W_h.weight.grad.abs().sum() > 0


def test_checkpoint_validation(facts, tmp_path):
    model = make_model(facts)
    state = model.state_dict()
    path = tmp_path / "model_best.tar"
    for wrapped in ("bare", "model", "state_dict"):
        payload = state if wrapped == "bare" else {wrapped: state, "optimizer_state_dict": {}, "epoch_id": 3}
        torch.save(payload, path)
        model.load_pretrained(path)
    for invalid in ({**state, "extra": torch.ones(1)}, dict(list(state.items())[1:]), KGICL({}).state_dict()):
        torch.save({"state_dict": invalid}, path)
        with pytest.raises(ValueError, match="architecture mismatch"):
            model.load_pretrained(path)


def test_invalid_settings_and_ids(facts):
    for args in ({"kgicl_dim": 0}, {"kgicl_shots": 0}, {"kgicl_query_batch_size": 0}, {"kgicl_prompt_seed": -1},
                 {"kgicl_masked_distances": [-1]}, {"normalization": "LayerNorm"}, {"byte_pair_encoding": True}):
        with pytest.raises(ValueError):
            KGICL(args)
    model = make_model(facts).eval()
    for triples in (torch.tensor([[6, 0, 1]]), torch.tensor([[0, 3, 1]]), torch.tensor([[-1, 0, 1]])):
        with pytest.raises(ValueError, match="vocabulary"):
            model(triples)
    with pytest.raises(ValueError):
        model.use_prompts({0: []})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("deterministic", [False, True])
def test_cuda_fused_path_matches_and_is_batch_invariant(facts, deterministic, monkeypatch):
    pytest.importorskip("triton")
    # PyTorch requires this for deterministic cuBLAS; the CQA Docker image sets it.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    model = KGICL(dict(num_entities=6, num_relations=3, kgicl_query_batch_size=4)).set_graph(facts).eval().requires_grad_(False)
    queries = torch.cat((facts[:, :2], torch.tensor([[5, 0], [1, 2]])))
    with torch.no_grad():
        cpu, cpu_heads = model(queries), model.forward_k_vs_all_heads(queries[:, [1, 0]])
    model.cuda()
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)
        for backend in ("torch", "triton"):
            model.set_inference_backend(backend)
            with torch.no_grad():
                batched = model(queries)
                model.query_batch_size = 1
                single = model(queries)
                reordered = model(queries.flip(0)).flip(0)
                model.query_batch_size = 4
                heads = model.forward_k_vs_all_heads(queries[:, [1, 0]].cuda())
            torch.testing.assert_close(batched.cpu(), cpu, atol=2e-4, rtol=2e-4)
            torch.testing.assert_close(heads.cpu(), cpu_heads, atol=2e-4, rtol=2e-4)
            # Triton rows never use atomics; PyTorch's CUDA scatter needs deterministic algorithms.
            if backend == "triton" or deterministic:
                assert torch.equal(batched, single) and torch.equal(batched, reordered)
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fused_hub_rows_and_query_blocks():
    """Hub rows are summed in fixed segments and query blocks are padded; a row never depends on its batch."""
    pytest.importorskip("triton")
    from dicee.models._triton_kgicl import HUB_SEGMENTS, QUERIES, SEGMENT
    generator = torch.Generator().manual_seed(0)
    n = 3 * SEGMENT * HUB_SEGMENTS

    def relations(count):
        return torch.randint(0, 3, (count,), generator=generator)
    # Entity 0 receives an edge from every other entity, a hub row of several segments.
    spokes = torch.stack((torch.arange(1, n), relations(n - 1), torch.zeros(n - 1, dtype=torch.long)), 1)
    noise = torch.stack((torch.randint(0, n, (4 * n,), generator=generator), relations(4 * n),
                         torch.randint(0, n, (4 * n,), generator=generator)), 1)
    model = KGICL(dict(num_entities=n, num_relations=3, kgicl_masked_distances=[0, 2], kgicl_query_batch_size=2 * QUERIES + 1))
    model = model.set_graph(torch.cat((spokes, noise))).eval().requires_grad_(False).cuda()
    queries = torch.cat((torch.tensor([[0, 0], [0, 1], [5, 2]]), noise[:10, :2])).cuda()
    with torch.no_grad():
        model.set_inference_backend("torch")
        expected = model(queries)
        model.set_inference_backend("triton")
        batched = model(queries)
        model.query_batch_size = 1
        single = model(queries)
        model.query_batch_size = 3
        reordered = model(queries.flip(0)).flip(0)
    torch.testing.assert_close(batched, expected, atol=2e-4, rtol=2e-4)
    assert torch.equal(batched, single) and torch.equal(batched, reordered)
    # Heads and entities first reached after two hops are masked.
    assert (batched[torch.arange(len(queries)), queries[:, 0]] == 0).all() and (batched != 0).any()


@pytest.mark.parametrize("model_name,technique,grouped,trainer", [
    ("KGICL", "NegSample", False, "torchCPUTrainer"),
    ("KGICL", "NegSample", True, "torchCPUTrainer"),
    ("KGICL", "KvsAll", False, "torchCPUTrainer"),
    ("KGICL", "1vsAll", False, "PL"),
])
def test_execute_and_reload(tmp_path, model_name, technique, grouped, trainer):
    from dicee.executer import Execute
    from dicee.knowledge_graph_embeddings import KGE
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "train.txt").write_text("a r b\na r c\nb s c\nc r d\nd s a\nb r d\n")
    (dataset / "valid.txt").write_text("a s d\n")
    (dataset / "test.txt").write_text("d r b\n")
    args = Namespace()
    args.model = model_name
    args.kgicl_dim, args.kgicl_attn_dim, args.kgicl_num_layers, args.kgicl_prompt_layers = 8, 3, 3, 2
    args.dataset_dir, args.path_to_store_single_run = str(dataset), str(tmp_path / "run")
    args.num_epochs, args.batch_size, args.neg_ratio, args.lr = 1, 2, 2, 0.001
    args.scoring_technique, args.trainer = technique, trainer
    args.backend, args.separator = "pandas", " "
    args.strict_negative_sampling = grouped
    args.adversarial_temperature = 1.0 if grouped else None
    if trainer == "PL":
        args.pl_trainer_kwargs = {"accelerator": "cpu", "devices": 1}
    execute = Execute(args)
    report = execute.start()
    assert np.isfinite(report["Test"]["MRR"])
    restored = KGE(path=str(tmp_path / "run"))
    triples = torch.tensor([[0, 0, 1]])
    with torch.no_grad():
        torch.testing.assert_close(restored.model(triples), execute.trained_model.eval()(triples))
    assert (tmp_path / "run" / "kgicl_graph.pt").is_file()
    restored.predict_missing_head_entity("r", "b")
    restored.predict_missing_tail_entity("a", "r")


def test_zero_epoch_official_checkpoint_format(tmp_path):
    from dicee.executer import Execute
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "train.txt").write_text("a r b\nb s c\nc r a\na s c\n")
    (dataset / "test.txt").write_text("a s b\n")
    path = tmp_path / "model_best.tar"
    pretrained = KGICL({})
    torch.save({"state_dict": pretrained.state_dict(), "optimizer_state_dict": {}, "epoch_id": 22}, path)
    args = Namespace()
    args.model, args.kgicl_checkpoint = "KGICL", str(path)
    args.dataset_dir, args.path_to_store_single_run = str(dataset), str(tmp_path / "run")
    args.num_epochs, args.batch_size = 0, 2
    args.scoring_technique, args.eval_model = "KvsAll", "test"
    report = Execute(args).start()
    state = torch.load(tmp_path / "run" / "model.pt", weights_only=True)
    for key in state:
        torch.testing.assert_close(state[key], pretrained.state_dict()[key])
    assert np.isfinite(report["Test"]["MRR"])
    assert json.loads((tmp_path / "run" / "configuration.json").read_text())["kgicl_dim"] == 32


def test_query_rows_are_bitwise_independent_of_batching(facts, monkeypatch):
    """The query evaluator caches rows across beams and controls; batching must not change a row."""
    from dicee.query_answering.context import QueryContext, attached_context
    from dicee.query_answering.engine import AtomicScorer
    from dicee.query_answering.method_evaluation import deterministic_kgfm
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    context = QueryContext([tuple(t) for t in facts.tolist()], 6, 3)
    conditions = [(h, r) for h in range(6) for r in range(3)]
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        model = KGICL(dict(num_entities=1, num_relations=1, kgicl_query_batch_size=4)).eval().requires_grad_(False).to(device)
        with deterministic_kgfm(), attached_context(model, context):
            reference = AtomicScorer(model, context, row_batch_size=1).rows(conditions)
            for size in (2, 5, 18):
                rows = AtomicScorer(model, context, row_batch_size=size).rows(conditions[::-1]).flip(0)
                assert torch.equal(rows, reference), (device, size)
        assert torch.isfinite(reference).all() and (reference != 0).any()
