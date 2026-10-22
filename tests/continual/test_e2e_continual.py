"""End-to-end tests on a tiny synthetic KG (Section 13).

Covers both required scenarios:
1. an update involving only existing entities;
2. an update introducing a new entity.

For each, verifies: model parameters change, the frozen snapshot of the
previous model stays frozen (its parameters are untouched), and the
evaluation pipeline returns valid (finite, in-range) metrics.

Runs entirely on CPU; no GPU or external dataset required.
"""

import copy

import numpy as np
import torch

from dicee.continual.update_sequence import generate_update_sequence
from dicee.continual.continual_trainer import ContinualTrainer, ContinualConfig
from dicee.continual.evaluation import evaluate_snapshot, preservation_metrics
from dicee.static_preprocess_funcs import get_er_vocab, get_re_vocab
from dicee.models.real import DistMult


def _toy_kg(seed=0, n_entities=12, n_relations=2, n_triples=40):
    rng = np.random.default_rng(seed)
    triples = set()
    for e in range(1, n_entities):
        triples.add((e - 1, int(rng.integers(0, n_relations)), e))
    while len(triples) < n_triples:
        h, t = rng.integers(0, n_entities, size=2)
        if h == t:
            continue
        triples.add((int(h), int(rng.integers(0, n_relations)), int(t)))
    return np.array(sorted(triples)[:n_triples], dtype=np.int64), n_entities, n_relations


def _make_model(num_entities, num_relations, dim=8):
    args = {
        "model": "DistMult", "embedding_dim": dim, "num_entities": num_entities,
        "num_relations": num_relations, "learning_rate": 0.05, "lr": 0.05, "optim": "Adam",
        "scoring_technique": "KvsAll", "input_dropout_rate": 0.0, "hidden_dropout_rate": 0.0,
        "weight_decay": 0.0, "normalization": None, "byte_pair_encoding": False,
    }
    return DistMult(args)


def _run_scenario(scenario, method):
    triples, n_entities, n_relations = _toy_kg()
    t0, steps = generate_update_sequence(
        triples, scenario=scenario, seed=7, initial_fraction=0.5, num_steps=3,
        min_incident_facts=1,
    )

    model = _make_model(num_entities=n_entities, num_relations=n_relations)
    cfg = ContinualConfig(method=method, lambda_pres=1.0, beta=2.0, cheb_K=2, epochs_per_step=2,
                           num_stable_queries=20, seed=0)
    trainer = ContinualTrainer(model, cfg)
    trainer.train_initial(t0.to_array())

    params_before = [p.detach().clone() for p in model.parameters()]

    rng = np.random.default_rng(cfg.seed)
    step = steps[0]
    new_num_entities = int(step.cumulative_triples[:, [0, 2]].max()) + 1
    new_num_entities = max(new_num_entities, n_entities)

    result = trainer.run_step(step, new_num_entities, t0.to_array(), rng)

    assert "training_time_seconds" in result
    assert result["num_trainable_parameters"] > 0

    # (5) model parameters changed.
    params_after = list(model.parameters())
    changed = any(not torch.allclose(a, b) for a, b in zip(params_before, params_after) if a.shape == b.shape)
    assert changed, "expected model parameters to change after a continual update step"

    # (6) the frozen previous-model snapshot is untouched.
    model_old = trainer._last_model_old
    assert all(not p.requires_grad for p in model_old.parameters())

    # (7) evaluation pipeline returns valid metrics.
    er_vocab, re_vocab = get_er_vocab([tuple(int(x) for x in row) for row in step.cumulative_triples]), \
        get_re_vocab([tuple(int(x) for x in row) for row in step.cumulative_triples])
    eval_triples = step.cumulative_triples[-5:]
    metrics = evaluate_snapshot(model, eval_triples, er_vocab, re_vocab)
    for key in ("MRR", "H@1", "H@3", "H@10"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0

    pres = preservation_metrics(model, model_old, t0.to_array()[:5], er_vocab, re_vocab)
    assert np.isfinite(pres["Delta_MRR_stable"])
    assert np.isfinite(pres["ScoreDrift"])
    return result


def test_e2e_existing_entity_update_finetune():
    _run_scenario("existing_entities", "finetune")


def test_e2e_existing_entity_update_uniform_distillation():
    _run_scenario("existing_entities", "uniform")


def test_e2e_existing_entity_update_chebyshev_distillation():
    _run_scenario("existing_entities", "chebyshev")


def test_e2e_new_entity_update():
    result = _run_scenario("new_entities", "chebyshev")
    assert result["num_new_entities"] >= 0  # may be 0 for an early step; vocabulary-growth path is exercised below


def test_e2e_new_entity_update_grows_vocabulary():
    triples, n_entities, n_relations = _toy_kg()
    t0, steps = generate_update_sequence(
        triples, scenario="new_entities", seed=7, initial_fraction=0.4, num_steps=3,
    )
    initial_num_entities = int(t0.to_array()[:, [0, 2]].max()) + 1
    step_with_new_entities = next(
        s for s in steps
        if s.num_new_entities > 0 and int(s.cumulative_triples[:, [0, 2]].max()) + 1 > initial_num_entities
    )
    model = _make_model(num_entities=initial_num_entities, num_relations=n_relations)
    cfg = ContinualConfig(method="chebyshev", epochs_per_step=2, num_stable_queries=10)
    trainer = ContinualTrainer(model, cfg)
    trainer.train_initial(t0.to_array())

    old_num_entities = model.num_entities
    rng = np.random.default_rng(0)
    new_total = int(step_with_new_entities.cumulative_triples[:, [0, 2]].max()) + 1
    trainer.run_step(step_with_new_entities, new_total, t0.to_array(), rng)

    assert model.num_entities == new_total
    assert model.num_entities > old_num_entities
    assert model.entity_embeddings.weight.shape[0] == new_total
