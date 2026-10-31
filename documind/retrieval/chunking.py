"""Split a contract into overlapping word chunks that keep their character positions.

Chunks are the retrieval units for BM25 and dense retrieval. Every chunk records where it sits in the
original text, so a retrieved chunk can be mapped back to a page and highlighted.
"""
import re

WORD_RE = re.compile(r"\S+")


def chunk_text(text, size=120, stride=60):
    """Chunks of `size` words, moving `stride` words each time (stride < size gives overlap)."""
    if stride <= 0 or stride > size:
        raise ValueError("stride must be between 1 and size")
    words = [(m.start(), m.end()) for m in WORD_RE.finditer(text)]
    if not words:
        return []
    chunks = []
    i = 0
    while True:
        j = min(i + size, len(words))
        start, end = words[i][0], words[j - 1][1]
        chunks.append({"char_start": start, "char_end": end, "text": text[start:end]})
        if j == len(words):
            break
        i += stride
    return chunks


def chunk_contains_answer(chunk, answers, min_fraction=0.5):
    """True if the chunk covers at least min_fraction of the shorter of (gold span, chunk) for some gold span.

    `answers` is a list of {"text", "start"} dicts.
    """
    chunk_len = chunk["char_end"] - chunk["char_start"]
    for a in answers:
        gold_start, gold_end = a["start"], a["start"] + len(a["text"])
        overlap = min(chunk["char_end"], gold_end) - max(chunk["char_start"], gold_start)
        if overlap > 0 and overlap >= min_fraction * min(gold_end - gold_start, chunk_len):
            return True
    return False