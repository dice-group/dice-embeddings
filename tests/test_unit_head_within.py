from types import MethodType, SimpleNamespace

import pytest
import torch

from dicee.knowledge_graph_embeddings import KGE


class HeadScorer:
    device = 'cpu'

    def __call__(self, triples):
        return triples[:, 0].float()


@pytest.mark.parametrize('inverse', [False, True])
@pytest.mark.parametrize('within', [['a'], ['c', 'a'], ['a', 'c'], None])
def test_head_candidate_restriction(inverse, within):
    kge = SimpleNamespace(
        all_have_inverse=inverse, model=HeadScorer(),
        entity_to_idx={'a': 0, 'b': 1, 'c': 2}, relation_to_idx={'r': 0},
        idx_to_entity={0: 'a', 1: 'b', 2: 'c'},
    )
    if inverse and within is None:
        kge.predict_missing_tail_entity = lambda *args: 'inverse path'
        assert KGE.predict_missing_head_entity(kge, 'r', 'a') == 'inverse path'
        return
    kge.predict_missing_head_entity = MethodType(KGE.predict_missing_head_entity, kge)
    candidates = within if within is not None else ['a', 'b', 'c']
    results = KGE.predict_topk(kge, r='r', t=['a', 'b'], within=within,
                               topk=len(candidates), batch_size=1)
    expected = sorted(candidates, key=kge.entity_to_idx.get, reverse=True)
    assert [[entity for entity, _ in row] for row in results] == [expected, expected]
    scores = kge.predict_missing_head_entity('r', ['a', 'b'], within=within)
    torch.testing.assert_close(
        scores, torch.tensor([float(kge.entity_to_idx[e]) for e in candidates] * 2),
    )
