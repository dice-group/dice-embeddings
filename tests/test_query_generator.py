import logging
import os
import pickle
import random

import pytest
import torch

from dicee import KGE, QueryGenerator


@pytest.fixture
def generator(tmp_path):
    """A small random KG written as tab-separated train/valid/test files."""
    rng = random.Random(0)
    entities = [f"e{i}" for i in range(30)]
    relations = [f"r{i}" for i in range(3)]
    triples = list({(rng.choice(entities), rng.choice(relations), rng.choice(entities)) for _ in range(500)})
    rng.shuffle(triples)
    n = len(triples)
    splits = {"train": triples[: int(0.7 * n)], "valid": triples[int(0.7 * n): int(0.85 * n)], "test": triples[int(0.85 * n):]}
    paths = {}
    for name, rows in splits.items():
        paths[name] = str(tmp_path / f"{name}.tsv")
        with open(paths[name], "w") as f:
            f.writelines(f"{h}\t{r}\t{t}\n" for h, r, t in rows)
    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: 2 * i for i, r in enumerate(relations)}  # even ids; odd ids are for inverse relations
    return QueryGenerator(paths["train"], paths["valid"], paths["test"], ent2id=ent2id, rel2id=rel2id, seed=1)


def test_every_query_has_hard_answers_disjoint_from_easy(generator):
    queries, easy, _, hard = generator.get_queries("2p", 10)
    flat = [q for qs in queries.values() for q in qs]
    assert flat
    for q in flat:
        assert hard[q]
        assert not (hard[q] & easy[q])


def test_queries_of_one_type_do_not_depend_on_earlier_types(generator):
    alone = generator.get_queries("2p", 10)[0]
    generator.get_queries("1p", 10)
    after_1p = generator.get_queries("2p", 10)[0]
    assert alone == after_1p


def test_save_queries_writes_pickles(generator, tmp_path):
    generator.save_queries("2i", 5, str(tmp_path / "out"))
    for suffix in ("queries", "easy-answers", "false-positives", "hard-answers"):
        with open(tmp_path / "out" / f"test-2i-{suffix}.pkl", "rb") as f:
            assert pickle.load(f) is not None
    assert len(os.listdir(tmp_path / "out")) == 4


def test_save_queries_rejects_unknown_type(generator, tmp_path):
    with pytest.raises(ValueError):
        generator.save_queries("not-a-type", 5, str(tmp_path))


class _StubKGE(KGE):
    """KGE whose atomic predictions are fixed raw logits, so no training is needed."""

    def __init__(self):
        self.entity_to_idx = {f"e{i}": i for i in range(3)}
        self._logits = {"a": torch.tensor([-3.0, -1.0, 2.0]), "b": torch.tensor([-2.0, -4.0, 3.0])}

    def predict(self, h=None, r=None, t=None, within=None, logits=True):
        scores = self._logits[r[0] if isinstance(r, list) else r]
        return scores if logits else torch.sigmoid(scores)


def test_intersection_defaults_to_probabilities_and_ranks_correctly():
    model = _StubKGE()
    query = (("e0", ("a",)), ("e0", ("b",)))
    top = model.answer_multi_hop_query(query_type="2i", query=query, tnorm="prod", k=1)
    assert top[0][0] == "e2"


def test_logits_with_prod_misrank_and_warn(caplog):
    model = _StubKGE()
    query = (("e0", ("a",)), ("e0", ("b",)))
    with caplog.at_level(logging.WARNING):
        scores = model.answer_multi_hop_query(query_type="2i", query=query, tnorm="prod", k=3, only_scores=True, use_logits=True)
    assert int(torch.argmax(scores)) != 2  # product of negative logits outranks the true best entity
    assert "use_logits=True" in caplog.text
