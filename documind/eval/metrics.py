"""Evaluation metrics for DocuMind AI.

All functions are pure and take plain lists or numpy arrays.
Metrics that are undefined (for example AUPR with no positive example) return None
instead of a made-up number.
"""
import re
import string
from collections import Counter

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize_text(text):
    """SQuAD-style normalization: lowercase, drop punctuation and articles, collapse spaces."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def token_f1(prediction, gold):
    """Token-overlap F1 between two strings. Both empty counts as a perfect match."""
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction, gold_spans):
    """F1 against the best-matching gold span. With no gold spans, only an empty prediction is right."""
    if not gold_spans:
        return 1.0 if not normalize_text(prediction) else 0.0
    return max(token_f1(prediction, g) for g in gold_spans)


def _has_positive(y_true):
    return np.asarray(y_true).sum() > 0


def aupr(y_true, scores):
    """Area under the precision-recall curve (average precision). None if there is no positive."""
    if not _has_positive(y_true):
        return None
    return float(average_precision_score(y_true, scores))


def precision_at_recall(y_true, scores, recall_target=0.8):
    """Best precision among thresholds that reach the target recall. None if there is no positive."""
    if not _has_positive(y_true):
        return None
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ok = recall >= recall_target
    return float(precision[ok].max())


def bootstrap_ci(y_true, scores, groups, metric_fn, n_boot=1000, seed=42, alpha=0.05):
    """Bootstrap confidence interval that resamples whole groups (contracts), not rows.

    Returns (point_estimate, lower, upper). Resamples where the metric is undefined
    (returns None) are skipped. Returns (point, None, None) if every resample was skipped.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    groups = np.asarray(groups)
    point = metric_fn(y_true, scores)

    unique_groups = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([index_by_group[g] for g in chosen])
        value = metric_fn(y_true[idx], scores[idx])
        if value is not None:
            values.append(value)
    if not values:
        return point, None, None
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)