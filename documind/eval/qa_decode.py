"""Turn raw QA model outputs (per-window start/end logits) into answers plus an uncertainty score.

Per window: the best non-null spans (start <= end, only context tokens, no two overlapping) and the null score,
which is start_logit[CLS] + end_logit[CLS] ("this window has no answer").
Per example (one contract and one category): the best span over all windows, plus a ranked list of
non-overlapping candidate spans, and
    present_score = best_span_score - null_score
where null_score is the MINIMUM null score over the windows (the SQuAD-v2 convention).
A higher present_score means the clause is more likely present. We abstain below a threshold.
"""
import numpy as np


def decode_window(start_logits, end_logits, offsets, n_best=20, max_answer_tokens=300, top_spans=5):
    """Best non-null spans and the null score for one window.

    `offsets` has one entry per token: (char_start, char_end) for context tokens, None otherwise
    (CLS, question, SEP). The logits must have the same length (no padding).
    Returns {"null_score", "span" (the best span or None), "spans" (up to top_spans non-overlapping spans, best first)}.
    """
    start_logits = np.asarray(start_logits, dtype=float)
    end_logits = np.asarray(end_logits, dtype=float)
    null_score = float(start_logits[0] + end_logits[0])
    ctx = np.array([i for i, o in enumerate(offsets) if o is not None], dtype=int)
    if len(ctx) == 0:
        return {"null_score": null_score, "span": None, "spans": []}

    top_starts = ctx[np.argsort(start_logits[ctx])[::-1][:n_best]]
    top_ends = ctx[np.argsort(end_logits[ctx])[::-1][:n_best]]
    candidates = []
    for s in top_starts:
        for e in top_ends:
            if e < s or (e - s + 1) > max_answer_tokens:
                continue
            candidates.append((float(start_logits[s] + end_logits[e]), int(s), int(e)))
    candidates.sort(key=lambda c: -c[0])

    chosen = []
    for score, s, e in candidates:
        if any(s <= ce and cs <= e for _, cs, ce in chosen):
            continue
        chosen.append((score, s, e))
        if len(chosen) == top_spans:
            break
    spans = [{"score": score, "token_start": s, "token_end": e,
              "char_start": int(offsets[s][0]), "char_end": int(offsets[e][1])} for score, s, e in chosen]
    return {"null_score": null_score, "span": spans[0] if spans else None, "spans": spans}


def aggregate_windows(window_results, context, k=5):
    """Combine the decoded windows of one example into a single prediction plus ranked candidates."""
    null_scores = [r["null_score"] for r in window_results]
    pool = []
    for w, result in enumerate(window_results):
        spans = result.get("spans")
        if spans is None:
            spans = [result["span"]] if result.get("span") is not None else []
        for sp in spans:
            pool.append((sp["score"], w, sp))
    pool.sort(key=lambda item: -item[0])  # stable: equal scores keep the earlier window first

    chosen = []
    for score, w, sp in pool:
        if any(sp["char_start"] < c["char_end"] and c["char_start"] < sp["char_end"] for _, _, c in chosen):
            continue  # same text found again by an overlapping window
        chosen.append((score, w, sp))
        if len(chosen) == k:
            break

    if not chosen or not null_scores:
        return {"text": "", "char_start": None, "char_end": None, "span_score": None,
                "null_score": min(null_scores) if null_scores else None,
                "present_score": float("-inf"), "window_index": None, "candidates": []}
    null_score = min(null_scores)
    candidates = [{
        "text": context[sp["char_start"]:sp["char_end"]],
        "char_start": sp["char_start"],
        "char_end": sp["char_end"],
        "span_score": score,
        "present_score": score - null_score,
        "window_index": w,
    } for score, w, sp in chosen]
    best = candidates[0]
    return {
        "text": best["text"],
        "char_start": best["char_start"],
        "char_end": best["char_end"],
        "span_score": best["span_score"],
        "null_score": null_score,
        "present_score": best["present_score"],
        "window_index": best["window_index"],
        "candidates": candidates,
    }