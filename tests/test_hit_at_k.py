import pytest

from documind.eval.hit_at_k import first_correct_rank, hit_flags, hit_report, per_category

GOLD = [{"text": "this agreement is effective as of january first two thousand twenty", "start": 10}]
RIGHT = {"text": "this agreement is effective as of january first two thousand twenty"}
WRONG = {"text": "payment is due within thirty days of the invoice date"}


def row(cands, title="C", category="Effective Date", answers=GOLD, score=10.0):
    return {"title": title, "category": category, "answers": answers, "present_score": score, "candidates": cands}


def test_rank_of_first_correct_candidate():
    assert first_correct_rank(row([RIGHT, WRONG])) == 1
    assert first_correct_rank(row([WRONG, WRONG, RIGHT])) == 3
    assert first_correct_rank(row([WRONG, WRONG])) is None
    assert first_correct_rank(row([])) is None


def test_hit_flags_nest_by_k():
    flags = hit_flags([row([WRONG, WRONG, RIGHT]), row([RIGHT]), row([WRONG])], ks=(1, 3, 5))
    assert flags[0] == {1: 0, 3: 1, 5: 1}
    assert flags[1] == {1: 1, 3: 1, 5: 1}
    assert flags[2] == {1: 0, 3: 0, 5: 0}


def test_hit_report_values_and_monotonic_in_k():
    rows = [row([WRONG, RIGHT], title=f"C{i}") for i in range(10)] + [row([RIGHT], title=f"D{i}") for i in range(10)]
    rep = hit_report(rows, ks=(1, 3, 5), n_boot=100, seed=1)
    assert rep["n"] == 20
    assert rep["hit@1"]["point"] == pytest.approx(0.5)
    assert rep["hit@3"]["point"] == pytest.approx(1.0)
    assert rep["hit@1"]["point"] <= rep["hit@3"]["point"] <= rep["hit@5"]["point"]
    assert rep["hit@1"]["lo"] <= rep["hit@1"]["point"] <= rep["hit@1"]["hi"]


def test_per_category_rates():
    rows = [row([RIGHT], category="A"), row([WRONG, RIGHT], category="A"), row([WRONG], category="B")]
    cats = per_category(rows, ks=(1, 3))
    assert cats["A"]["n_present"] == 2 and cats["A"]["hit@1"] == pytest.approx(0.5) and cats["A"]["hit@3"] == pytest.approx(1.0)
    assert cats["B"]["hit@3"] == pytest.approx(0.0)


def test_missing_candidates_gives_a_clear_error():
    with pytest.raises(ValueError):
        first_correct_rank({"answers": GOLD, "pred_text": "x"})