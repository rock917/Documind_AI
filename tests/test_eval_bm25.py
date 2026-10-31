import pytest

from documind.retrieval.eval_bm25 import build_query, first_relevant_rank, random_recall_at_k


def test_build_query_uses_category_and_details():
    q = 'Highlight the parts (if any) of this contract related to "Governing Law" that should be reviewed by a lawyer. Details: Which state law governs?'
    assert build_query("Governing Law", q) == "Governing Law Which state law governs?"
    assert build_query("Term", "no details marker here") == "Term"


def test_first_relevant_rank():
    relevant = [False, True, False, True]
    assert first_relevant_rank([2, 0, 3, 1], relevant) == 3
    assert first_relevant_rank([1, 0, 2, 3], relevant) == 1
    assert first_relevant_rank([0, 2], relevant) is None


def test_random_recall_matches_hand_computation():
    # 10 chunks, 1 relevant, top 3: chance 3/10
    assert random_recall_at_k(10, 1, 3) == pytest.approx(0.3)
    # 10 chunks, 2 relevant, top 1: 2/10
    assert random_recall_at_k(10, 2, 1) == pytest.approx(0.2)
    # top k covers every non-relevant chunk: certain
    assert random_recall_at_k(5, 3, 3) == pytest.approx(1.0)
    assert random_recall_at_k(5, 0, 3) == 0.0