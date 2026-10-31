"""Sliding-window features with labels for extractive QA (SQuAD-v2 style, with abstention).

make_windows() turns one (question, contract, gold answers) into windows and labels each:
  positive: an answer lies fully inside the window. start/end are token positions of the
            earliest such answer.
  partial : an answer overlaps the window but none fits fully. Excluded from training later,
            because calling it "no answer" would teach the model to ignore real clause text.
  negative: no gold answer overlaps the window. start = end = 0 (the CLS token).
"""

POSITIVE = "positive"
PARTIAL = "partial"
NEGATIVE = "negative"


def _token_span(offsets, first_ctx, last_ctx, start_char, end_char):
    """Token positions (start, end) of the char span, or None if it does not map cleanly."""
    i = first_ctx
    while i <= last_ctx and offsets[i][0] <= start_char:
        i += 1
    start_tok = i - 1
    j = last_ctx
    while j >= first_ctx and offsets[j][1] >= end_char:
        j -= 1
    end_tok = j + 1
    if start_tok < first_ctx or end_tok > last_ctx or start_tok > end_tok:
        return None
    return start_tok, end_tok


def make_windows(tok, question, context, answers, max_length=384, stride=128):
    """Return a list of window dicts. `answers` is a list of {"text", "start"} dicts."""
    enc = tok(
        question,
        context,
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )
    spans = sorted((a["start"], a["start"] + len(a["text"])) for a in answers)
    windows = []
    for w in range(len(enc["input_ids"])):
        seq = enc.sequence_ids(w)
        raw_offsets = enc["offset_mapping"][w]
        ctx_idx = [k for k, s in enumerate(seq) if s == 1]
        first_ctx, last_ctx = ctx_idx[0], ctx_idx[-1]
        win_start, win_end = raw_offsets[first_ctx][0], raw_offsets[last_ctx][1]

        label, start_pos, end_pos = NEGATIVE, 0, 0
        overlaps_any = False
        for s, t in spans:
            if s < win_end and t > win_start:
                overlaps_any = True
            if win_start <= s and t <= win_end:
                mapped = _token_span(raw_offsets, first_ctx, last_ctx, s, t)
                if mapped is not None:
                    label, (start_pos, end_pos) = POSITIVE, mapped
                    break
        if label != POSITIVE and overlaps_any:
            label = PARTIAL

        windows.append({
            "window_index": w,
            "input_ids": enc["input_ids"][w],
            "token_type_ids": enc["token_type_ids"][w],
            "attention_mask": enc["attention_mask"][w],
            "offset_mapping": [tuple(o) if seq[k] == 1 else None for k, o in enumerate(raw_offsets)],
            "start_position": start_pos,
            "end_position": end_pos,
            "label": label,
        })
    return windows