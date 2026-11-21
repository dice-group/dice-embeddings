"""The score input of the sigmoid calibrations: raw logits (default, unchanged) or each row's log-softmax."""

import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from dicee.query_answering import AdapterQuery, AdapterTrainingData, QueryAnswerer, QueryContext, QueryScoreAdapter, fit_query_adapter
from dicee.query_answering.adapter_training import _bank, _prepared_bank, _query_logs
from dicee.query_answering.context import state_fingerprint
from dicee.query_answering.score_adapter import score_features


class Rows(nn.Module):
    """Rows z[h, r] over 7 entities, shifted by a per-row constant ``offsets[h, r]`` (default zero).

    The values are multiples of 1/1024 and the offsets integers below 256, so every shifted float32 row is exact.
    """

    num_entities, num_relations = 7, 4

    def __init__(self, offsets=None):
        super().__init__()
        self.values = nn.Parameter(torch.round(torch.randn(7, 4, 7, generator=torch.Generator().manual_seed(8)) * 1024) / 1024)
        self.register_buffer('offsets', torch.zeros(7, 4) if offsets is None else torch.as_tensor(offsets, dtype=torch.float32))

    def forward_k_vs_all(self, conditions):
        return self.values[conditions[:, 0], conditions[:, 1]] + self.offsets[conditions[:, 0], conditions[:, 1], None]


CONTEXT = QueryContext([(0, 0, 1), (0, 0, 2), (2, 1, 6), (1, 1, 3), (3, 2, 4), (4, 3, 5)], 7, 4)
CONDITIONS = [(h, r) for h in range(7) for r in range(4)]
WEIGHTS = torch.randn(2, 8, generator=torch.Generator().manual_seed(9)).double() / 5


def rows_and_context(model=None):
    model = Rows() if model is None else model
    with torch.no_grad():
        raw = model.forward_k_vs_all(torch.tensor(CONDITIONS))
    observed, base = CONTEXT.features(CONDITIONS, device='cpu')
    return raw, observed, base


@pytest.mark.parametrize('mode', ['global', 'context_scores'])
@pytest.mark.parametrize('normalization', ['none', 'standard'])
def test_raw_input_is_the_earlier_computation_and_artifact(mode, normalization, tmp_path):
    raw, observed, base = rows_and_context()
    weights = WEIGHTS if mode == 'context_scores' else WEIGHTS[:, :1]
    default = QueryScoreAdapter(mode, 1., bias_bound=8., weights=weights, normalization=normalization)
    explicit = QueryScoreAdapter(mode, 1., bias_bound=8., weights=weights, normalization=normalization, score_input='raw')
    prepared, _, features = default.prepare(raw, observed, base)
    # The earlier prepare: float64 rows, features of those rows, then the optional standardization.
    expected = raw.to(torch.float64)
    torch.testing.assert_close(features, score_features(expected, observed, base, mode), rtol=0, atol=0)
    if normalization == 'none':
        assert torch.equal(prepared, expected)
    assert torch.equal(default(raw, observed, base), explicit(raw, observed, base))
    # The artifact keeps version 3 and no score_input field, so files and cache keys are unchanged.
    assert 'score_input' not in default.configuration and default.to_dict()['version'] == 3
    assert default.to_dict() == explicit.to_dict()


def test_log_softmax_input_equals_a_hand_computation():
    raw = torch.tensor([[95.5, 97.25, 101., 99.75], [-3., 0.5, 2., -1.]], dtype=torch.float32)
    observed, base = QueryContext([(0, 0, 2), (0, 0, 3), (1, 0, 0)], 4, 1).features([(0, 0), (1, 0)], device='cpu')
    adapter = QueryScoreAdapter('context_scores', 1., bias_bound=8., weights=WEIGHTS, score_input='log_softmax')
    # z - logsumexp(z), written out with the standard library.
    rows = raw.double().tolist()
    z = np.array([[v - (max(row) + math.log(sum(math.exp(u - max(row)) for u in row))) for v in row] for row in rows])
    scores = adapter.scores(raw)
    np.testing.assert_allclose(scores.numpy(), z, rtol=0, atol=1e-12)
    # The same float64 operation as the link-prediction harness of ProofPFN.
    assert torch.equal(scores, torch.log_softmax(raw.double(), 1))
    # Features and the sigmoid read the transformed rows.
    prob = np.exp(z)
    top = np.sort(z, axis=1)[:, -2:]
    obs = observed.numpy()
    features = np.column_stack((base.numpy(), np.tanh(z.mean(1) / 4), -(prob * np.log(prob)).sum(1) / np.log(4),
                                np.tanh((top[:, 1] - top[:, 0]) / 4), np.tanh(((z * obs).sum(1) / obs.sum(1) - z.mean(1)) / 4)))
    weights = WEIGHTS.numpy()
    logits = 2 ** np.tanh(features @ weights[0])[:, None] * z + 8 * np.tanh(features @ weights[1])[:, None]
    expected = 1 / (1 + np.exp(-logits))
    expected[obs] = 1.
    np.testing.assert_allclose(adapter(raw, observed, base).detach().exp().numpy(), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize('mode', ['global', 'context_scores'])
def test_log_softmax_input_ignores_a_constant_per_row(mode):
    offsets = torch.randint(-200, 200, (7, 4), generator=torch.Generator().manual_seed(3)).float()
    weights = WEIGHTS if mode == 'context_scores' else WEIGHTS[:, :1]
    raw, observed, base = rows_and_context()
    shifted, _, _ = rows_and_context(Rows(offsets))
    for score_input in ('log_softmax', 'raw'):
        adapter = QueryScoreAdapter(mode, 0., bias_bound=8., weights=weights, score_input=score_input)
        before, after = adapter(raw, observed, base), adapter(shifted, observed, base)
        if score_input == 'log_softmax':
            torch.testing.assert_close(after, before, rtol=0, atol=1e-9)
        else:
            assert not torch.allclose(after, before, atol=1e-3)


def test_log_softmax_artifacts_round_trip_and_refuse_ambiguous_files(tmp_path):
    model = Rows()
    adapter = QueryScoreAdapter('context_scores', 1., bias_bound=8., weights=WEIGHTS, score_input='log_softmax',
                                metadata={'backbone_state_sha256': state_fingerprint(model)})
    payload = adapter.to_dict()
    # Version 4 marks the new input, so readers without score_input refuse the file instead of scoring raw logits.
    assert payload['version'] == 4 and payload['score_input'] == 'log_softmax'
    adapter.save(tmp_path / 'adapter.json')
    loaded = QueryScoreAdapter.load(tmp_path / 'adapter.json', model=model)
    assert loaded.score_input == 'log_softmax' and loaded.configuration == adapter.configuration
    raw, observed, base = rows_and_context(model)
    assert torch.equal(loaded(raw, observed, base), adapter(raw, observed, base))
    for change in (dict(version=3), dict(score_input='raw'), dict(score_input='softmax')):
        (tmp_path / 'changed.json').write_text(json.dumps(dict(payload, **change)))
        with pytest.raises(ValueError):
            QueryScoreAdapter.load(tmp_path / 'changed.json', model=model)
    with pytest.raises(ValueError, match='score_input'):
        QueryScoreAdapter('global', score_input='softmax')
    with pytest.raises(ValueError, match='Fixed calibrations'):
        QueryScoreAdapter('global', 1., fixed_calibration='softmax-degree', score_input='log_softmax')


def test_queries_and_engine_caches_use_the_score_input():
    model = Rows().eval()
    raw_adapter = QueryScoreAdapter('context_scores', 1., bias_bound=8., weights=WEIGHTS)
    log_adapter = QueryScoreAdapter('context_scores', 1., bias_bound=8., weights=WEIGHTS, score_input='log_softmax')
    query = ((0, (0,)), (1, (1,)))
    raw_engine = QueryAnswerer(model, context=CONTEXT, adapter=raw_adapter, row_batch_size=2)
    log_engine = QueryAnswerer(model, context=CONTEXT, adapter=log_adapter, row_batch_size=2)
    raw_scores = raw_engine.predict(query, return_log_scores=True)
    log_scores = log_engine.predict(query, return_log_scores=True)
    assert not torch.equal(raw_scores, log_scores)
    raw, observed, base = rows_and_context(model)
    expected = log_adapter(raw[[0, 5]], observed[[0, 5]], base[[0, 5]]).sum(0)
    torch.testing.assert_close(log_scores, expected, rtol=0, atol=1e-14)


def source_data(shapes):
    """A source whose training queries are flat intersections, optionally with a 1p validation query."""
    train = (AdapterQuery(((0, (0,)), (1, (1,))), {2, 3}), AdapterQuery(((3, (2,)), (4, (3,))), {4, 5}))
    validation = (AdapterQuery((0, (0,)), {1, 2, 5}),) if '1p' in shapes else ()
    return AdapterTrainingData('source', CONTEXT, train, validation)


@pytest.mark.parametrize('shapes', [('2i',), ('2i', '1p')])
def test_fitting_applies_the_score_input_like_inference(shapes, tmp_path):
    """Score banks keep raw rows; the fitted adapter scores training queries exactly as the evaluation scores them."""
    model = Rows().eval()
    data = source_data(shapes)
    result = fit_query_adapter(model, [data], epochs=3, score_input='log_softmax', cache_dir=tmp_path / 'banks',
                               bias_bound=8., row_batch_size=2)
    adapter = result.adapter
    assert adapter.score_input == 'log_softmax' and adapter.to_dict()['version'] == 4
    bank = _bank(model, data, cache_dir=tmp_path / 'banks', row_batch_size=2, seed=0, samples=None)
    try:
        prepared = _prepared_bank(adapter, bank)
        engine = QueryAnswerer(model, context=data.context, adapter=adapter, row_batch_size=2)
        for query in (*data.train, *data.validation):
            torch.testing.assert_close(_query_logs(adapter, prepared, query), engine.predict(query.query, return_log_scores=True),
                                       rtol=0, atol=1e-12)
    finally:
        if 'provider' in bank:
            bank['provider'].close()
    # Shifting every row by a constant changes nothing when the adapter reads log-softmax rows.
    shifted = Rows(torch.randint(-50, 50, (7, 4), generator=torch.Generator().manual_seed(5)).float()).eval()
    with torch.no_grad():
        shifted.values.copy_(model.values)
    again = fit_query_adapter(shifted, [data], epochs=3, score_input='log_softmax', bias_bound=8., row_batch_size=2)
    torch.testing.assert_close(again.adapter.weights, adapter.weights, rtol=0, atol=1e-9)
