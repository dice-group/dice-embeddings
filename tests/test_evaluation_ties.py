"""Tie semantics, seeded evaluation, and policy routing without model training."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dicee.config import Namespace
from dicee.evaluation import link_prediction as lp
from dicee.evaluation._filtering import (
    TIE_POLICIES,
    FilteredRanker,
    compute_filtered_rank,
    compute_filtered_rank_batch,
    evaluation_tie_options,
)
from dicee.evaluation.ensemble import evaluate_ensemble_link_prediction_performance
from dicee.evaluation.evaluator import Evaluator
from dicee.knowledge_graph_embeddings import KGE
from dicee.scripts.run import get_default_arguments
from dicee.static_preprocess_funcs import preprocesses_input_args


@pytest.mark.parametrize('policy,expected', [('optimistic', 1), ('pessimistic', 3)])
def test_filtered_tie_bounds(policy, expected):
    scores = torch.tensor([9., 7., 7., 7., 6., 7.])
    original = scores.clone()
    assert compute_filtered_rank(scores, 2, [0, 1, 2, 1], tie_policy=policy) == expected
    assert torch.equal(scores, original)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.uint32, np.uint64, np.int32, np.int64])
def test_cpu_batched_bounds_accept_numpy_filter_ids(dtype):
    # KG vocabularies retain NumPy's compact unsigned entity IDs. They must
    # become integer indices, never byte masks or unsupported unsigned tensors.
    scores = torch.tensor([[9., 7., 7., 7., 6., 7.], [1., 1., 2., 1., 1., 0.]])
    targets = np.array([2, 0], dtype=dtype)
    filters = [list(np.array([0, 1, 2, 1], dtype=dtype)), list(np.array([0, 2, 4], dtype=dtype))]
    ranker = FilteredRanker('pessimistic')
    assert ranker.bounds_batch(scores, targets, filters) == [(1, 2), (1, 2)]
    assert ranker.rank_batch(scores, targets, filters) == [3, 3]
    assert ranker.bounds_batch(scores, targets, filters, row_indices=[0, 1]) == [(1, 2), (1, 2)]


@pytest.mark.parametrize('policy', TIE_POLICIES)
def test_unique_scores_keep_same_rank(policy):
    assert compute_filtered_rank(torch.tensor([4., 3., 2., 1.]), 2, [0, 2],
                                 tie_policy=policy) == 2


@pytest.mark.parametrize('policy', ['optimistic', 'random', 'pessimistic'])
def test_filtered_negative_infinity_is_not_a_tie(policy):
    assert compute_filtered_rank(torch.tensor([-torch.inf, -torch.inf, 1.]), 0,
                                 [0, 1], tie_policy=policy) == 2
    assert compute_filtered_rank(torch.ones(4), 2, [0, 1, 2, 3], tie_policy=policy) == 1
    # Numerically close scores are not rounded into a tie.
    assert compute_filtered_rank(torch.tensor([1., 1.000001]), 0, [], tie_policy=policy) == 2


def test_random_is_uniform_repeatable_and_independent_of_global_rng():
    def sample(seed):
        ranker = FilteredRanker('random', seed)
        return [ranker.rank(torch.ones(6), 0, [0, 4, 5]) for _ in range(6000)]
    before = torch.random.get_rng_state().clone()
    ranks = sample(123)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert ranks == sample(123)
    assert ranks != sample(124)
    counts = np.bincount(ranks, minlength=5)[1:]
    assert len(counts) == 4
    assert np.all(abs(counts - 1500) < 150), counts


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_legacy_sort_order_and_random_cpu_gpu_parity(device):
    scores = torch.zeros((3, 513), device=device)
    targets = torch.tensor([0, 16, 512])
    filters = [[0, 2], [16, 2], [512, 2]]
    masked = scores.clone()
    masked[:, 2] = -torch.inf
    order = torch.sort(masked, dim=1, descending=True).indices
    expected = [torch.where(row == t)[0].item() + 1 for row, t in zip(order, targets)]
    assert FilteredRanker().rank_batch(scores, targets, filters) == expected
    for i, target in enumerate(targets):
        old_order = torch.sort(masked[i], descending=True).indices
        old_rank = torch.where(old_order == target)[0].item() + 1
        assert compute_filtered_rank(scores[i], target, filters[i]) == old_rank
    ranks = compute_filtered_rank_batch(scores, targets, filters, tie_policy='random', tie_seed=9)
    assert ranks == compute_filtered_rank_batch(scores.cpu(), targets, filters,
                                                tie_policy='random', tie_seed=9)


def test_policy_validation_and_config_defaults():
    assert evaluation_tie_options(SimpleNamespace()) == {'tie_policy': 'sort', 'tie_seed': 0}
    assert Namespace().eval_tie_policy == 'sort'
    assert get_default_arguments([]).eval_tie_policy == 'sort'
    args = get_default_arguments(['--eval_tie_policy', 'random', '--random_seed', '7'])
    assert evaluation_tie_options(args) == {'tie_policy': 'random', 'tie_seed': 7}
    args.eval_tie_seed = 42
    assert evaluation_tie_options(vars(args))['tie_seed'] == 42
    for policy in TIE_POLICIES:
        assert get_default_arguments(['--eval_tie_policy', policy]).eval_tie_policy == policy
    with pytest.raises(ValueError, match='Unknown tie policy'):
        Evaluator(SimpleNamespace(eval_tie_policy='invalid'))
    with pytest.raises(ValueError, match='Unknown tie policy'):
        preprocesses_input_args(SimpleNamespace(eval_tie_policy='invalid'))
    with pytest.raises(SystemExit):
        get_default_arguments(['--eval_tie_policy', 'invalid'])


@pytest.mark.parametrize('policy', ['optimistic', 'random', 'pessimistic'])
def test_reject_unfiltered_nan(policy):
    with pytest.raises(ValueError, match='NaN'):
        compute_filtered_rank(torch.tensor([1., float('nan')]), 0, [], tie_policy=policy)
    assert compute_filtered_rank(torch.tensor([1., float('nan')]), 0, [1], tie_policy=policy) == 1


class TiedModel(torch.nn.Module):
    """Constant scores, with six candidates and both triple/KvsAll interfaces."""
    def __init__(self):
        super().__init__()
        self.args = {'batch_size': 3}
        self.name = 'TiedModel'
        self.str_to_bpe_entity_to_idx = {f'e{i}': i for i in range(6)}

    def forward(self, x):
        return torch.ones(len(x)) if x.shape[1] == 3 else torch.ones(len(x), 6)

    def forward_triples(self, x):
        return torch.ones(len(x))

    def forward_k_vs_all(self, x):
        return torch.ones(len(x), 6)


class TiedGraphModel(TiedModel):
    def forward_k_vs_all_heads(self, x):
        return torch.ones(len(x), 6)


class WrappedModel:
    def __init__(self):
        self.model = TiedModel()
        self.num_entities = 6
        self.entity_to_idx = {f'e{i}': i for i in range(6)}
        self.relation_to_idx = {'r': 0}

    def __call__(self, x):
        return self.model(x)

    def get_entity_index(self, entity):
        return self.entity_to_idx[entity]

    def get_relation_index(self, relation):
        return self.relation_to_idx[relation]

    def get_bpe_token_representation(self, value):
        if isinstance(value, list):
            return [self.get_bpe_token_representation(v) for v in value]
        i = 7 if value == 'r' else self.entity_to_idx[value]
        return (i, i)


ROUTES = ['native', 'graph', 'evaluator_neg', 'entity', 'relation', 'constrained',
          'ensemble', 'wrapper', 'reciprocals', 'bpe_wrapper', 'bpe_reciprocals',
          'bpe_native', 'bpe_kvsall', 'bpe_evaluator', 'bpe_evaluator_neg', 'kge']


@pytest.mark.parametrize('route', ROUTES)
@pytest.mark.parametrize('policy', ['optimistic', 'pessimistic', 'random'])
def test_policy_reaches_every_evaluator(route, policy):
    triples = np.array([[0, 0, 1]] * 12)
    er, re = {(0, 0): [1, 3]}, {(0, 1): [0, 3]}
    string_triples = [['e0', 'r', 'e1']] * 12
    string_er, string_re = {('e0', 'r'): ['e1', 'e3']}, {('r', 'e1'): ['e0', 'e3']}
    model, wrapper = TiedModel(), WrappedModel()
    bpe = wrapper.get_bpe_token_representation
    entities = list(wrapper.entity_to_idx)
    opts = dict(tie_policy=policy, tie_seed=42)
    evaluator = Evaluator(SimpleNamespace(batch_size=3, eval_model='test',
                                         eval_tie_policy=policy, eval_tie_seed=42))
    evaluator.num_entities = 6
    evaluator.er_vocab, evaluator.re_vocab = er, re
    evaluator.ee_vocab = {(0, 1): [0, 3]}
    evaluator.func_triple_to_bpe_representation = lambda triple: [bpe(v) for v in triple]
    bpe_triples = [tuple(bpe(v) for v in triple) for triple in string_triples]
    bpe_er = {(bpe('e0'), bpe('r')): [bpe('e1'), bpe('e3')]}
    bpe_re = {(bpe('r'), bpe('e1')): [bpe('e0'), bpe('e3')]}
    bidirectional = route in ['native', 'graph', 'evaluator_neg', 'wrapper', 'bpe_wrapper',
                              'bpe_native', 'bpe_evaluator_neg', 'kge']
    if route in ('native', 'graph'):
        def run():
            return lp.evaluate_lp(TiedGraphModel() if route == 'graph' else model, triples, 6, er, re, batch_size=3, **opts)
    elif route == 'evaluator_neg':
        def run():
            return evaluator.evaluate_lp(model, triples, 'ties')
    elif route in ('entity', 'relation', 'constrained'):
        if route == 'constrained':
            evaluator.args.eval_model = 'test_constraint'
            evaluator.er_vocab = {(0, 0): [1]}
            evaluator.range_constraints_per_rel = {0: [1, 3]}
        label = 'RelationPrediction' if route == 'relation' else 'EntityPrediction'
        def run():
            return evaluator.evaluate_lp_k_vs_all(model, triples, form_of_labelling=label)
    elif route == 'ensemble':
        def run():
            return evaluate_ensemble_link_prediction_performance([model, model], triples, er, weighted_averaging=False, batch_size=3, **opts)
    elif route == 'wrapper':
        def run():
            return lp.evaluate_link_prediction_performance(wrapper, string_triples, string_er, string_re, **opts)
    elif route == 'reciprocals':
        def run():
            return lp.evaluate_link_prediction_performance_with_reciprocals(wrapper, string_triples, string_er, **opts)
    elif route == 'bpe_wrapper':
        def run():
            return lp.evaluate_link_prediction_performance_with_bpe(wrapper, entities, string_triples, string_er, string_re, **opts)
    elif route == 'bpe_reciprocals':
        def run():
            return lp.evaluate_link_prediction_performance_with_bpe_reciprocals(wrapper, entities, string_triples, string_er, **opts)
    elif route == 'bpe_native':
        def run():
            return lp.evaluate_bpe_lp(model, bpe_triples, [bpe(e) for e in entities], bpe_er, bpe_re, **opts)
    elif route == 'bpe_kvsall':
        def run():
            return lp.evaluate_lp_bpe_k_vs_all(wrapper, string_triples, string_er, 3, evaluator.func_triple_to_bpe_representation, wrapper.entity_to_idx, **opts)
    elif route == 'bpe_evaluator':
        evaluator.er_vocab = string_er
        def run():
            return evaluator.evaluate_lp_bpe_k_vs_all(model, string_triples)
    elif route == 'bpe_evaluator_neg':
        evaluator.er_vocab, evaluator.re_vocab = bpe_er, bpe_re
        def run():
            evaluator.eval_rank_of_head_and_tail_byte_pair_encoded_entity(
                test_set=bpe_triples, ordered_bpe_entities=[bpe(e) for e in entities], trained_model=model)
            return evaluator.report['Test']
    elif route == 'kge':
        wrapper.configs = {'eval_tie_policy': policy, 'eval_tie_seed': 42}
        # No positives removed in unfiltered mode, so use all six tied candidates.
        def run():
            return KGE.eval_lp_performance(wrapper, string_triples, filtered=False)
    candidates = 6 if route == 'kge' else 5
    count = len(triples) * (2 if bidirectional else 1)
    if policy == 'random':
        expected_ranks = torch.randint(candidates, (count,), generator=torch.Generator().manual_seed(42)) + 1
    else:
        expected_ranks = torch.full((count,), 1 if policy == 'optimistic' else candidates)
    expected = {'MRR': expected_ranks.double().reciprocal().mean().item(),
                **{f'H@{k}': (expected_ranks <= k).double().mean().item() for k in (1, 3, 10)}}
    assert run() == pytest.approx(expected)
    assert run() == pytest.approx(expected)  # A split evaluation restarts its tie RNG.


def test_random_stream_can_be_resumed_across_native_evaluation_batches():
    triples = np.array([[0, 0, 1]] * 13)
    er, re = {(0, 0): [1, 3]}, {(0, 1): [0, 3]}
    args = (TiedGraphModel(), triples, 6, er, re)
    expected = lp.evaluate_lp(*args, tie_policy='random', tie_seed=42, batch_size=3)
    generator = torch.Generator().manual_seed(42)
    first = lp.evaluate_lp(args[0], triples[:5], 6, er, re, tie_policy='random', tie_generator=generator)
    state = generator.get_state().tolist()  # Same JSON representation as the benchmark runner.
    resumed = torch.Generator().set_state(torch.tensor(state, dtype=torch.uint8))
    second = lp.evaluate_lp(args[0], triples[5:], 6, er, re, tie_policy='random', tie_generator=resumed)
    assert {key: (first[key] * 5 + second[key] * 8) / 13 for key in expected} == pytest.approx(expected)


def test_loaded_kge_inherits_saved_policy_and_accepts_overrides(tmp_path):
    import pickle
    wrapper = WrappedModel()
    wrapper.path = str(tmp_path)
    wrapper.configs = {'eval_tie_policy': 'pessimistic', 'eval_tie_seed': 42}
    for name, vocab in [('er_vocab', {(0, 0): [1, 3]}), ('re_vocab', {(0, 1): [0, 3]})]:
        (tmp_path / f'{name}.p').write_bytes(pickle.dumps(vocab))
    triples = [['e0', 'r', 'e1']]
    assert KGE.eval_lp_performance(wrapper, triples)['MRR'] == pytest.approx(0.2)
    assert KGE.eval_lp_performance(wrapper, triples, tie_policy='optimistic')['MRR'] == 1


def test_kgfm_runner_restores_random_state_after_interruption(tmp_path, monkeypatch):
    """Exercise the actual progress writer/resumer on a synthetic KG and model."""
    import importlib.util
    import json
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / 'benchmarks/kgfm_zero_shot.py'
    spec = importlib.util.spec_from_file_location('kgfm_tie_test', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    dataset = tmp_path / 'synthetic'
    dataset.mkdir()
    for split in ('train', 'valid', 'test'):
        (dataset / f'{split}.txt').write_text('e0 r e1\n')
    checkpoint = tmp_path / 'dummy-checkpoint'
    checkpoint.write_bytes(b'synthetic')
    facts = np.array([[3, 0, 4]])
    kg = SimpleNamespace(train_set=facts, valid_set=np.array([[2, 0, 5]]),
                         test_set=np.array([[0, 0, 1]] * 13), num_entities=6, num_relations=1,
                         er_vocab={(0, 0): [1, 3]}, re_vocab={(0, 1): [0, 3]})

    class FrozenTiedModel(TiedGraphModel):
        def __init__(self, settings):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.edge_type = torch.zeros(2, dtype=torch.long)

        def load_pretrained(self, checkpoint):
            return self

        def set_graph(self, facts):
            return self

    monkeypatch.setitem(runner.MODELS, 'ULTRA', FrozenTiedModel)
    # Keep the checkpoint under ROOT so the real metadata builder accepts it.
    monkeypatch.setitem(runner.CHECKPOINTS, 'ULTRA', 'checkpoints/ultra_3g.pth')
    original_digest = runner.digest
    monkeypatch.setattr(runner, 'digest', lambda p: original_digest(checkpoint)
                        if p.name == 'ultra_3g.pth' else original_digest(p))
    monkeypatch.setattr(runner, 'KG', lambda **kwargs: kg)
    args = [str(path), '--model', 'ULTRA', '--dataset', str(dataset), '--device', 'cpu',
            '--threads', str(torch.get_num_threads()), '--batch-size', '3',
            '--tie-policy', 'random', '--tie-seed', '42']
    output = tmp_path / 'resumed'
    monkeypatch.setattr(sys, 'argv', args + ['--output', str(output)])
    original_evaluate = runner.evaluate_lp
    calls = 0

    def interrupt_second_batch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('Simulated interruption')
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(runner, 'evaluate_lp', interrupt_second_batch)
    with pytest.raises(RuntimeError, match='Simulated interruption'):
        runner.main()
    progress = json.loads((output / 'progress.json').read_text())
    assert progress['completed'] == 3
    assert progress['tie_rng_state']
    monkeypatch.setattr(runner, 'evaluate_lp', original_evaluate)
    runner.main()
    resumed = json.loads((output / 'result.json').read_text())
    monkeypatch.setattr(sys, 'argv', args + ['--output', str(tmp_path / 'uninterrupted')])
    runner.main()
    uninterrupted = json.loads((tmp_path / 'uninterrupted/result.json').read_text())
    assert resumed['metrics'] == uninterrupted['metrics']
    assert resumed['ranked_queries'] == 26
    assert resumed['configuration']['tie_policy'] == 'random'
    assert resumed['configuration']['tie_seed'] == 42
