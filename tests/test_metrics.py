import numpy as np
import pytest

from documind.eval.metrics import (
    aupr,
    best_f1,
    bootstrap_ci,
    normalize_text,
    precision_at_recall,
    token_f1,
)


def test_normalize_text():
    assert normalize_text("The Governing Law.") == "governing law"
    assert normalize_text("  A   term,  of ONE year ") == "term of one year"


def test_token_f1_identical_and_disjoint():
    assert token_f1("delaware law applies", "Delaware law applies.") == 1.0
    assert token_f1("alpha beta", "gamma delta") == 0.0


def test_token_f1_partial_overlap():
    # precision 2/3, recall 1.0 -> F1 = 0.8
    assert token_f1("delaware law applies", "delaware law") == pytest.approx(0.8)


def test_token_f1_empty_cases():
    assert token_f1("", "") == 1.0
    assert token_f1("something", "") == 0.0
    assert token_f1("", "something") == 0.0


def test_best_f1_takes_max_over_gold_spans():
    golds = ["one year", "delaware law applies"]
    assert best_f1("delaware law applies", golds) == 1.0


def test_best_f1_no_gold_spans():
    assert best_f1("", []) == 1.0        # correctly says "not found"
    assert best_f1("some text", []) == 0.0  # found something that is not there


def test_aupr_perfect_and_known_value():
    assert aupr([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(1.0)
    # precision 1.0 at recall 0.5, then 2/3 at recall 1.0 -> 0.5*1 + 0.5*(2/3)
    assert aupr([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(5 / 6)


def test_aupr_undefined_without_positives():
    assert aupr([0, 0, 0], [0.1, 0.2, 0.3]) is None


def test_precision_at_recall():
    y, s = [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]
    assert precision_at_recall(y, s, 0.5) == pytest.approx(1.0)
    assert precision_at_recall(y, s, 0.8) == pytest.approx(2 / 3)


def test_precision_at_recall_undefined_without_positives():
    assert precision_at_recall([0, 0], [0.1, 0.2], 0.8) is None


def test_bootstrap_ci_is_deterministic_and_ordered():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 5)
    y = rng.integers(0, 2, size=150)
    s = y * 0.5 + rng.random(150)
    a = bootstrap_ci(y, s, groups, aupr, n_boot=200, seed=1)
    b = bootstrap_ci(y, s, groups, aupr, n_boot=200, seed=1)
    assert a == b
    point, lo, hi = a
    assert lo <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_bootstrap_ci_perfect_ranking_has_zero_width():
    groups = np.repeat(np.arange(10), 4)
    y = np.tile([1, 1, 0, 0], 10)
    s = np.tile([0.9, 0.8, 0.2, 0.1], 10)
    point, lo, hi = bootstrap_ci(y, s, groups, aupr, n_boot=100, seed=3)
    assert point == pytest.approx(1.0)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_bootstrap_resamples_whole_contracts_not_rows():
    # Group 0 holds every positive. Any resample without group 0 has no positives,
    # so those resamples must be skipped, not counted as 0.
    groups = np.array([0, 0, 1, 1, 2, 2])
    y = np.array([1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.1])
    point, lo, hi = bootstrap_ci(y, s, groups, aupr, n_boot=200, seed=5)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)