import numpy as np
import pytest

from documind.eval.risk_coverage import (
    area_under_risk_coverage,
    correct_flags,
    operating_point,
    point_at_threshold,
    risk_coverage_curve,
)


def test_perfect_ranking_has_zero_risk_until_all_correct_are_output():
    scores = [9, 8, 7, 1, 0]
    correct = [1, 1, 1, 0, 0]
    curve = risk_coverage_curve(scores, correct, n_answerable=3)
    assert curve["risk"][:3].tolist() == [0.0, 0.0, 0.0]
    assert curve["risk"][4] == pytest.approx(2 / 5)
    assert curve["recall"][2] == pytest.approx(1.0)
    assert curve["coverage"][-1] == pytest.approx(1.0)


def test_area_is_lower_for_better_ranking():
    good = risk_coverage_curve([9, 8, 1, 0], [1, 1, 0, 0], 2)
    bad = risk_coverage_curve([9, 8, 1, 0], [0, 0, 1, 1], 2)
    assert area_under_risk_coverage(good) < area_under_risk_coverage(bad)


def test_operating_point_picks_largest_coverage_under_risk_limit():
    curve = risk_coverage_curve([9, 8, 7, 1, 0], [1, 1, 1, 0, 0], 3)
    op = operating_point(curve, 0.0)
    assert op["coverage"] == pytest.approx(3 / 5)
    assert op["answer_if_score_at_least"] == pytest.approx(7)
    assert op["recall"] == pytest.approx(1.0)


def test_operating_point_none_when_unreachable():
    curve = risk_coverage_curve([9, 8], [0, 0], 1)
    assert operating_point(curve, 0.1) is None


def test_point_at_threshold_uses_strict_greater_than():
    scores = np.array([9.0, 8.0, 7.0, 1.0])
    correct = np.array([1, 1, 0, 0])
    p = point_at_threshold(scores, correct, n_answerable=2, threshold=7.0)
    assert p["n_outputs"] == 2 and p["coverage"] == pytest.approx(0.5)
    assert p["risk"] == pytest.approx(0.0) and p["recall"] == pytest.approx(1.0)
    empty = point_at_threshold(scores, correct, 2, 100.0)
    assert empty["n_outputs"] == 0 and empty["risk"] is None


def test_correct_flags_require_present_clause_and_f1_match():
    rows = [
        {"answers": [{"text": "delaware law applies", "start": 0}], "pred_text": "delaware law applies"},
        {"answers": [{"text": "delaware law applies", "start": 0}], "pred_text": "something unrelated here"},
        {"answers": [], "pred_text": "delaware law applies"},
    ]
    assert correct_flags(rows).tolist() == [1, 0, 0]