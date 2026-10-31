import math

import numpy as np
import pytest

from documind.eval.qa_decode import aggregate_windows, decode_window

# token 0 = CLS, 1-2 = question, 3 = SEP, 4-7 = context words
OFFSETS = [None, None, None, None, (0, 5), (6, 10), (11, 16), (17, 20)]
CONTEXT = "alpha beta gamma pi!"


def logits(size=8, **peaks):
    arr = np.zeros(size)
    for idx, value in peaks.items():
        arr[int(idx[1:])] = value
    return arr


def test_decode_picks_best_context_span_and_maps_to_chars():
    start = logits(i5=4.0)
    end = logits(i6=3.0)
    out = decode_window(start, end, OFFSETS)
    assert out["span"]["token_start"] == 5 and out["span"]["token_end"] == 6
    assert (out["span"]["char_start"], out["span"]["char_end"]) == (6, 16)
    assert out["span"]["score"] == pytest.approx(7.0)
    assert out["null_score"] == pytest.approx(0.0)


def test_decode_ignores_question_and_special_tokens():
    start = logits(i1=100.0, i5=2.0)  # question token has the biggest logit
    end = logits(i2=100.0, i6=2.0)
    out = decode_window(start, end, OFFSETS)
    assert out["span"]["token_start"] == 5 and out["span"]["token_end"] == 6


def test_decode_requires_start_before_end():
    start = logits(i7=5.0)
    end = logits(i4=5.0)  # best pair is invalid (end before start)
    out = decode_window(start, end, OFFSETS)
    assert out["span"]["token_start"] <= out["span"]["token_end"]


def test_decode_respects_max_answer_tokens():
    start = logits(i4=5.0)
    end = logits(i7=5.0)
    long = decode_window(start, end, OFFSETS, max_answer_tokens=10)
    short = decode_window(start, end, OFFSETS, max_answer_tokens=2)
    assert (long["span"]["token_start"], long["span"]["token_end"]) == (4, 7)
    assert short["span"]["token_end"] - short["span"]["token_start"] + 1 <= 2


def test_decode_null_score_uses_cls_logits():
    out = decode_window(logits(i0=3.0), logits(i0=2.0), OFFSETS)
    assert out["null_score"] == pytest.approx(5.0)


def test_window_without_context_tokens_has_no_span():
    out = decode_window(np.zeros(4), np.zeros(4), [None, None, None, None])
    assert out["span"] is None


def test_aggregate_takes_best_span_and_minimum_null_score():
    a = {"null_score": 2.0, "span": {"score": 5.0, "char_start": 0, "char_end": 5}}
    b = {"null_score": -1.0, "span": {"score": 9.0, "char_start": 6, "char_end": 10}}
    out = aggregate_windows([a, b], CONTEXT)
    assert out["text"] == "beta"
    assert out["window_index"] == 1
    assert out["null_score"] == pytest.approx(-1.0)
    assert out["present_score"] == pytest.approx(10.0)


def test_aggregate_with_no_windows_is_safely_empty():
    out = aggregate_windows([], CONTEXT)
    assert out["text"] == ""
    assert math.isinf(out["present_score"]) and out["present_score"] < 0


def test_decode_returns_non_overlapping_spans_best_first():
    start = logits(i4=5.0, i6=4.0)
    end = logits(i5=5.0, i7=4.0)
    out = decode_window(start, end, OFFSETS, top_spans=2)
    assert [(s["token_start"], s["token_end"]) for s in out["spans"]] == [(4, 5), (6, 7)]
    assert out["spans"][0]["score"] == pytest.approx(10.0)
    assert out["spans"][1]["score"] == pytest.approx(8.0)
    assert out["span"] == out["spans"][0]


def test_decode_respects_top_spans_limit_and_never_overlaps():
    out = decode_window(logits(i4=5.0, i5=4.0, i6=3.0, i7=2.0), logits(i4=1.0, i5=2.0, i6=3.0, i7=4.0), OFFSETS, top_spans=3)
    assert len(out["spans"]) <= 3
    for a in out["spans"]:
        for b in out["spans"]:
            if a is not b:
                assert a["token_end"] < b["token_start"] or b["token_end"] < a["token_start"]


def test_aggregate_drops_duplicate_spans_from_overlapping_windows():
    a = {"null_score": 0.0, "spans": [{"score": 9.0, "char_start": 0, "char_end": 10}]}
    b = {"null_score": 1.0, "spans": [{"score": 7.0, "char_start": 5, "char_end": 12},
                                      {"score": 6.0, "char_start": 20, "char_end": 30}]}
    out = aggregate_windows([a, b], "x" * 40, k=5)
    assert [(c["char_start"], c["char_end"]) for c in out["candidates"]] == [(0, 10), (20, 30)]
    assert out["candidates"][1]["window_index"] == 1
    assert out["candidates"][1]["present_score"] == pytest.approx(6.0 - 0.0)


def test_aggregate_candidates_respect_k_and_top1_matches_summary_fields():
    spans = [{"score": 9.0 - i, "char_start": 10 * i, "char_end": 10 * i + 5} for i in range(4)]
    out = aggregate_windows([{"null_score": 1.0, "spans": spans}], "y" * 60, k=2)
    assert len(out["candidates"]) == 2
    assert out["text"] == out["candidates"][0]["text"]
    assert out["present_score"] == pytest.approx(out["candidates"][0]["present_score"])
    assert out["present_score"] == pytest.approx(9.0 - 1.0)