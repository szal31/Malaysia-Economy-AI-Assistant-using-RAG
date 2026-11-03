import pytest

from evaluate import hit_at_k, reciprocal_rank

RANKED = [("a", 1), ("b", 2)]


def test_hit_at_k():
    assert hit_at_k(RANKED, {("b", 2)}, k=1) == 0
    assert hit_at_k(RANKED, {("b", 2)}, k=2) == 1


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, {("b", 2)}) == 0.5
    assert reciprocal_rank(RANKED, {("z", 9)}) == 0


def test_empty_gold_rejected():
    with pytest.raises(ValueError):
        hit_at_k(RANKED, set())
