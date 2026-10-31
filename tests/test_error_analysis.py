import pytest

from documind.eval.error_analysis import analyse, classify_present, guard_not_test


def row(answers, pred_text, char_start, char_end, score=10.0, category="Gov Law", title="C1"):
    return {"title": title, "category": category, "answers": answers, "has_answer": bool(answers),
            "pred_text": pred_text, "char_start": char_start, "char_end": char_end, "present_score": score}


GOLD = [{"text": "this agreement is governed by delaware law and the courts of new york", "start": 100}]


def test_hit_when_text_matches():
    r = row(GOLD, GOLD[0]["text"], 100, 100 + len(GOLD[0]["text"]))
    assert classify_present(r) == "hit"


def test_wrong_place_when_no_overlap():
    r = row(GOLD, "payment is due within thirty days of the invoice", 500, 548)
    assert classify_present(r) == "wrong_place"


def test_overlap_too_short_and_too_long():
    short = row(GOLD, "delaware", 130, 138)
    assert classify_present(short) == "overlap_too_short"
    long_text = "x " * 400
    long = row(GOLD, long_text, 90, 90 + len(long_text))
    assert classify_present(long) == "overlap_too_long"


def test_no_prediction():
    assert classify_present(row(GOLD, "", None, None)) == "no_prediction"


def test_analyse_counts_types_and_false_alarms():
    rows = [
        row(GOLD, GOLD[0]["text"], 100, 100 + len(GOLD[0]["text"]), score=12.0),
        row(GOLD, "payment is due within thirty days of the invoice", 500, 548, score=11.0),
        row([], "some random text", 5, 21, score=9.0, category="Term"),   # false alarm
        row([], "other text", 7, 17, score=1.0, category="Term"),         # correctly abstained
    ]
    rep = analyse(rows, threshold=8.0)
    assert rep["present_error_types"]["hit"] == 1
    assert rep["present_error_types"]["wrong_place"] == 1
    assert rep["false_alarms_at_threshold"] == 1
    assert rep["false_alarm_rate_among_absent"] == pytest.approx(0.5)
    assert rep["false_alarms_by_category"] == {"Term": 1}
    assert rep["per_category"]["Gov Law"]["hit_rate"] == pytest.approx(0.5)


def test_guard_refuses_test_files():
    with pytest.raises(ValueError):
        guard_not_test("data/predictions/model_test.jsonl")
    guard_not_test("data/predictions/model_val.jsonl")