import pytest

from documind.retrieval.chunking import chunk_contains_answer, chunk_text


def words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_empty_text_has_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_chunks_slice_the_original_text_exactly():
    text = "alpha  beta\ngamma delta   epsilon zeta " * 20
    for c in chunk_text(text, size=10, stride=5):
        assert text[c["char_start"]:c["char_end"]] == c["text"]
        assert not c["text"].startswith(" ") and not c["text"].endswith(" ")


def test_chunk_sizes_overlap_and_coverage():
    text = words(100)
    chunks = chunk_text(text, size=30, stride=10)
    assert len(chunks[0]["text"].split()) == 30
    assert chunks[1]["text"].split()[0] == "w10"      # stride moves by 10 words
    assert chunks[-1]["text"].split()[-1] == "w99"     # the last word is covered
    assert chunks[0]["char_start"] == 0


def test_short_text_is_one_chunk():
    chunks = chunk_text(words(5), size=30, stride=10)
    assert len(chunks) == 1 and chunks[0]["text"] == words(5)


def test_invalid_stride_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("a b c", size=5, stride=0)
    with pytest.raises(ValueError):
        chunk_text("a b c", size=5, stride=6)


def test_chunk_contains_answer_rules():
    chunk = {"char_start": 100, "char_end": 300}
    inside = [{"text": "x" * 50, "start": 150}]
    far = [{"text": "x" * 50, "start": 1000}]
    barely = [{"text": "x" * 100, "start": 290}]           # only 10 of 100 chars overlap
    half = [{"text": "x" * 100, "start": 250}]             # 50 of 100 chars overlap
    long_gold = [{"text": "x" * 1000, "start": 0}]         # chunk (200 chars) fully inside a long gold span
    assert chunk_contains_answer(chunk, inside)
    assert not chunk_contains_answer(chunk, far)
    assert not chunk_contains_answer(chunk, barely)
    assert chunk_contains_answer(chunk, half)
    assert chunk_contains_answer(chunk, long_gold)
    assert not chunk_contains_answer(chunk, [])