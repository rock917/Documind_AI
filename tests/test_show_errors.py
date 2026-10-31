import pytest

from documind.eval.show_errors import select_errors

GOLD = [{"text": "this agreement is effective as of january first two thousand twenty", "start": 100}]


def row(category, answers, pred, cs, ce, score):
    return {"title": "C", "category": category, "answers": answers, "has_answer": bool(answers),
            "pred_text": pred, "char_start": cs, "char_end": ce, "present_score": score}


def test_selects_only_wrong_spans_of_the_category_most_confident_first():
    rows = [
        row("Effective Date", GOLD, GOLD[0]["text"], 100, 100 + len(GOLD[0]["text"]), 20.0),   # hit
        row("Effective Date", GOLD, "an unrelated sentence about payment terms", 900, 940, 5.0),
        row("Effective Date", GOLD, "another unrelated sentence entirely different", 800, 845, 9.0),
        row("Agreement Date", GOLD, "an unrelated sentence about payment terms", 900, 940, 30.0),  # other category
        row("Effective Date", [], "something", 5, 14, 40.0),                                      # absent clause
    ]
    errors = select_errors(rows, "Effective Date")
    assert [round(r["present_score"]) for _, r in errors] == [9, 5]
    assert all(t == "wrong_place" for t, _ in errors)


def test_none_scores_sort_last():
    rows = [
        row("Effective Date", GOLD, "unrelated words here again", 900, 925, None),
        row("Effective Date", GOLD, "different unrelated words", 700, 724, 1.0),
    ]
    errors = select_errors(rows, "Effective Date")
    assert errors[0][1]["present_score"] == 1.0 and errors[1][1]["present_score"] is None