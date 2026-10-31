import random

from documind.data.build_windows import select_windows


def fake(labels):
    return [{"label": lab, "window_index": i} for i, lab in enumerate(labels)]


def test_positives_always_kept_partials_always_dropped():
    wins = fake(["positive", "partial", "negative", "positive"])
    kept = select_windows(wins, random.Random(0), True, 0.0, 0.0)
    assert [w["label"] for w in kept] == ["positive", "positive"]


def test_probability_one_keeps_all_negatives_and_zero_keeps_none():
    wins = fake(["negative"] * 20)
    assert len(select_windows(wins, random.Random(0), True, 1.0, 0.0)) == 20
    assert len(select_windows(wins, random.Random(0), True, 0.0, 1.0)) == 0


def test_hard_probability_for_answered_and_easy_for_empty_examples():
    wins = fake(["negative"] * 20)
    assert len(select_windows(wins, random.Random(0), False, 0.0, 1.0)) == 20
    assert len(select_windows(wins, random.Random(0), False, 1.0, 0.0)) == 0


def test_sampling_is_deterministic_for_a_seed():
    wins = fake(["negative"] * 200)
    a = select_windows(wins, random.Random(7), True, 0.3, 0.3)
    b = select_windows(wins, random.Random(7), True, 0.3, 0.3)
    assert [w["window_index"] for w in a] == [w["window_index"] for w in b]
    assert 30 < len(a) < 90