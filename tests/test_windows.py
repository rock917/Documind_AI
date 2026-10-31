import pytest
from transformers import BertTokenizerFast

from documind.data.windows import NEGATIVE, PARTIAL, POSITIVE, make_windows

WORDS = [f"w{i}" for i in range(60)]


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    path = tmp_path_factory.mktemp("vocab") / "vocab.txt"
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS + ["highlight", "clause"]
    path.write_text("\n".join(vocab), encoding="utf-8")
    return BertTokenizerFast(str(path), do_lower_case=True)


QUESTION = "highlight clause"


def contract(n):
    return " ".join(WORDS[i % len(WORDS)] for i in range(n))


def answer(ctx, first_word, n_words):
    words = ctx.split(" ")
    start = len(" ".join(words[:first_word])) + (1 if first_word else 0)
    text = " ".join(words[first_word:first_word + n_words])
    assert ctx[start:start + len(text)] == text
    return {"text": text, "start": start}


def decoded_span(win, ctx):
    s, e = win["start_position"], win["end_position"]
    return ctx[win["offset_mapping"][s][0]:win["offset_mapping"][e][1]]


def test_short_contract_single_positive_window_roundtrips(tok):
    ctx = contract(20)
    ans = answer(ctx, 5, 3)
    wins = make_windows(tok, QUESTION, ctx, [ans], max_length=64, stride=16)
    assert len(wins) == 1
    assert wins[0]["label"] == POSITIVE
    assert decoded_span(wins[0], ctx) == ans["text"]


def test_no_answers_means_all_windows_negative_with_cls_target(tok):
    ctx = contract(200)
    wins = make_windows(tok, QUESTION, ctx, [], max_length=32, stride=8)
    assert len(wins) > 1
    assert all(w["label"] == NEGATIVE for w in wins)
    assert all(w["start_position"] == 0 and w["end_position"] == 0 for w in wins)


def test_long_contract_positive_roundtrip_and_targets_are_context_tokens(tok):
    ctx = contract(200)
    ans = answer(ctx, 100, 4)
    wins = make_windows(tok, QUESTION, ctx, [ans], max_length=32, stride=8)
    positives = [w for w in wins if w["label"] == POSITIVE]
    assert positives, "the answer should fit fully inside at least one window"
    for w in positives:
        assert decoded_span(w, ctx) == ans["text"]
        assert w["offset_mapping"][w["start_position"]] is not None  # never a question token
        assert w["offset_mapping"][w["end_position"]] is not None
    for w in wins:
        if w["label"] == NEGATIVE:
            assert w["start_position"] == 0 and w["end_position"] == 0


def test_answer_longer_than_any_window_is_never_positive(tok):
    ctx = contract(200)
    ans = answer(ctx, 50, 80)  # far longer than the context room in a 32-token window
    wins = make_windows(tok, QUESTION, ctx, [ans], max_length=32, stride=8)
    labels = {w["label"] for w in wins}
    assert POSITIVE not in labels
    assert PARTIAL in labels
    assert NEGATIVE in labels


def test_partial_window_is_not_labelled_negative(tok):
    ctx = contract(200)
    ans = answer(ctx, 40, 10)
    wins = make_windows(tok, QUESTION, ctx, [ans], max_length=32, stride=8)
    partial = [w for w in wins if w["label"] == PARTIAL]
    assert partial, "a boundary-crossing answer should produce partial windows"
    assert all(w["start_position"] == 0 and w["end_position"] == 0 for w in partial)


def test_earliest_answer_wins_when_several_fit_in_one_window(tok):
    ctx = contract(20)
    late = answer(ctx, 12, 2)
    early = answer(ctx, 3, 2)
    wins = make_windows(tok, QUESTION, ctx, [late, early], max_length=64, stride=16)
    assert len(wins) == 1 and wins[0]["label"] == POSITIVE
    assert decoded_span(wins[0], ctx) == early["text"]


def test_window_fields_are_aligned(tok):
    ctx = contract(120)
    wins = make_windows(tok, QUESTION, ctx, [], max_length=32, stride=8)
    for w in wins:
        n = len(w["input_ids"])
        assert n <= 32
        assert len(w["attention_mask"]) == n == len(w["offset_mapping"]) == len(w["token_type_ids"])