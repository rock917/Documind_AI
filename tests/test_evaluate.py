import pytest

from documind.eval.evaluate import (
    choose_threshold,
    compute_report,
    f1_at_threshold,
    guard_threshold,
)
import numpy as np


def row(title, category, has_answer, score, pred="the clause text"):
    answers = [{"text": "the clause text", "start": 0}] if has_answer else []
    return {"title": title, "category": category, "answers": answers, "has_answer": has_answer,
            "pred_text": pred, "present_score": score}


def perfect_rows(n_contracts=12):
    rows = []
    for c in range(n_contracts):
        rows.append(row(f"C{c}", "Gov Law", True, 5.0 + c * 0.01))
        rows.append(row(f"C{c}", "Term", False, -5.0 - c * 0.01))
    return rows


def test_choose_threshold_separates_the_classes():
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    f1_span = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    f1_abstain = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    t = choose_threshold(scores, f1_span, f1_abstain)
    assert 3.0 <= t < 4.0
    assert f1_at_threshold(f1_span, f1_abstain, scores, t) == pytest.approx(1.0)


def test_perfect_predictions_give_perfect_scores():
    report = compute_report(perfect_rows(), n_boot=50, seed=1)
    assert report["micro_aupr"]["point"] == pytest.approx(1.0)
    assert report["micro_precision_at_recall_0.8"]["point"] == pytest.approx(1.0)
    assert report["f1_with_abstention"]["point"] == pytest.approx(1.0)
    assert report["answerable_examples"]["said_present_rate"] == pytest.approx(1.0)
    assert report["unanswerable_examples"]["correctly_abstained_rate"] == pytest.approx(1.0)
    assert report["threshold_source"] == "chosen_on_this_file"


def test_always_abstain_baseline_equals_fraction_without_answer():
    rows = perfect_rows()
    rows += [row(f"D{i}", "Term", False, -9.0) for i in range(12)]  # 12 + 12 empty of 36
    report = compute_report(rows, n_boot=20, seed=1)
    assert report["baseline_always_abstain_f1"] == pytest.approx(24 / 36)
    assert report["aupr_chance_level"] == pytest.approx(12 / 36)


def test_given_threshold_is_used_and_labelled():
    report = compute_report(perfect_rows(), threshold=100.0, n_boot=20, seed=1)
    assert report["threshold"] == 100.0 and report["threshold_source"] == "given"
    # threshold above every score: always abstain, so F1 equals the always-abstain baseline
    assert report["f1_with_abstention"]["point"] == pytest.approx(report["baseline_always_abstain_f1"])


def test_category_without_positives_has_undefined_aupr_and_is_listed():
    rows = perfect_rows()
    assert compute_report(rows, n_boot=20, seed=1)["per_category"]["Term"]["aupr"] is None
    report = compute_report(rows, n_boot=20, seed=1)
    assert report["categories_without_positives"] == ["Term"]
    assert report["macro_aupr_n_categories"] == 1


def test_missing_present_score_is_treated_as_lowest():
    rows = perfect_rows()
    rows[0]["present_score"] = None
    report = compute_report(rows, n_boot=20, seed=1)
    assert report["n_examples"] == len(rows)


def test_guard_refuses_tuning_on_test_file_but_allows_given_threshold():
    with pytest.raises(ValueError):
        guard_threshold("data/predictions/model_test.jsonl", None)
    guard_threshold("data/predictions/model_test.jsonl", 1.5)
    guard_threshold("data/predictions/model_val.jsonl", None)