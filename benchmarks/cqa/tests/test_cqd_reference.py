"""Optional independent CQD oracle tests.

Set DICEE_CQD_REFERENCE to the pinned upstream checkout (d1ce74164936a7c09d9147e83190da047cb39429).
No DICE package or repository-local model fixtures are needed. Tolerance here
quantifies partition rounding against the dense upstream profile; it does not
change the exported profile's score/ranking/tie verification gates.
"""

import importlib.util
import inspect
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

QUERIES = {'1p': [0, 0], '2p': [0, 0, 2], '3p': [0, 0, 2, 1], '4p': [0, 0, 2, 1, 3],
           '2i': [0, 0, 1, 2], '3i': [0, 0, 1, 2, 2, 1], '4i': [0, 0, 1, 2, 2, 1, 3, 3],
           'ip': [0, 0, 1, 2, 2], 'pi': [0, 0, 2, 1, 2], '2u': [0, 0, 1, 2, -1], 'up': [0, 0, 1, 2, -1, 2]}


@pytest.fixture(scope='module')
def reference():
    checkout = os.environ.get('DICEE_CQD_REFERENCE')
    if not checkout:
        pytest.skip('Set DICEE_CQD_REFERENCE to the optional pinned CQD checkout')
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.syspath_prepend(checkout)
        from cqd import discrete
        from cqd.base import CQD
        path = Path(__file__).resolve().parents[1] / 'verification' / 'reference_cqd.py'
        spec = importlib.util.spec_from_file_location('bounded_cqd_reference', path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        yield CQD, discrete, helper


@pytest.mark.parametrize('hybrid', [False, True])
@pytest.mark.parametrize('tnorm', ['prod', 'min'])
@pytest.mark.parametrize('batch_size', [2, 32])
def test_positive_shapes_against_upstream(reference, hybrid, tnorm, batch_size):
    CQD, source, helper = reference
    filters = {(0, 0): [1, 2], (1, 2): [3], (2, 2): [4], (3, 0): [5], (5, 2): [6], (4, 0): [0]}
    model = CQD(11, 4, 7, k=3, max_k=5, max_norm=.9 if hybrid else 1., filters=filters,
                do_normalize=True, t_norm_name=tnorm).eval()
    generator = torch.Generator().manual_seed(107)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))

        def scoring(lhs, relation, rhs):
            values = model.score_o(lhs, relation, rhs)[0]
            return model.max_norm * (values - values.min()) / (values.max() - values.min())

        for shape, flat in QUERIES.items():
            queries = torch.tensor([flat])
            function = getattr(source, 'query_' + {'2u': '2u_dnf', 'up': 'up_dnf'}.get(shape, shape))
            arguments = dict(entity_embeddings=model.embeddings[0], predicate_embeddings=model.embeddings[1],
                             queries=queries, filters=filters, k=model.k, max_k=model.max_k, max_norm=model.max_norm,
                             scoring_function=scoring, t_norm=torch.mul if tnorm == 'prod' else torch.minimum,
                             t_conorm=(lambda a, b: 1 - (1 - a) * (1 - b)) if tnorm == 'prod' else torch.maximum)
            expected = function(**{name: arguments[name] for name in inspect.signature(function).parameters})
            with patch.object(model, 'score_o', wraps=model.score_o) as calls:
                actual = helper.query_bounded(model, source, shape, queries, batch_size)
            assert all(call.args[0].shape[0] <= batch_size for call in calls.call_args_list)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6, msg=(shape, hybrid, tnorm, batch_size))
            assert torch.equal(actual, helper.query_bounded(model, source, shape, queries, batch_size))


def test_rejects_multiple_query_batches(reference):
    _, source, helper = reference
    with pytest.raises(ValueError, match='one query'):
        helper.query_bounded(None, source, '1p', torch.zeros(2, 2, dtype=torch.long), 32)
