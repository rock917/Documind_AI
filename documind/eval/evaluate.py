"""Evaluate saved predictions (from documind.eval.predict) with bootstrap confidence intervals.

Reports, over all (contract, category) pairs in the file:
  - AUPR and precision at 80% recall of the "clause is present" score, with 95% CIs that resample whole contracts
  - SQuAD-v2 style F1 with abstention at a threshold, plus the trivial "always say not found" baseline
  - per-category AUPR (None when a category has no positive example) and the macro mean
The threshold is chosen on the file you evaluate unless you pass --threshold. Choose it on VAL only,
then pass it when evaluating TEST; tuning on the test file is refused.

Run from the project root (E:\\Documind) with the virtual environment active:
    python -m documind.eval.evaluate --pred data\\predictions\\legalbert_qa_val.jsonl
Writes reports\\eval_<file name>.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from documind.eval.metrics import aupr, best_f1, bootstrap_ci, precision_at_recall

NEG_INF = -1e9


def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def guard_threshold(pred_path, threshold):
    """Refuse to tune a threshold on a test predictions file."""
    if threshold is None and str(pred_path).lower().endswith("_test.jsonl"):
        raise ValueError("Refusing to choose a threshold on the test predictions. "
                         "Choose it on val, then pass --threshold.")


def per_example_f1(rows):
    """F1 if we answer with the predicted span, and F1 if we abstain, for every example."""
    f1_span, f1_abstain = [], []
    for r in rows:
        golds = [a["text"] for a in r["answers"]]
        f1_span.append(best_f1(r["pred_text"], golds))
        f1_abstain.append(1.0 if not golds else 0.0)
    return np.array(f1_span), np.array(f1_abstain)


def f1_at_threshold(f1_span, f1_abstain, scores, threshold):
    return float(np.where(scores > threshold, f1_span, f1_abstain).mean())


def choose_threshold(scores, f1_span, f1_abstain, max_candidates=1000):
    """Threshold on the present score that maximizes mean F1 (answer if score > threshold)."""
    unique = np.unique(scores)
    if len(unique) > max_candidates:
        unique = np.unique(np.quantile(scores, np.linspace(0, 1, max_candidates)))
    candidates = np.concatenate([[scores.min() - 1.0], unique])
    values = [f1_at_threshold(f1_span, f1_abstain, scores, t) for t in candidates]
    return float(candidates[int(np.argmax(values))])


def ci_dict(result):
    point, lo, hi = result
    return {"point": point, "lo": lo, "hi": hi}


def compute_report(rows, threshold=None, n_boot=1000, seed=42):
    y = np.array([1 if r["has_answer"] else 0 for r in rows])
    scores = np.array([r["present_score"] if r["present_score"] is not None else NEG_INF for r in rows], dtype=float)
    groups = np.array([r["title"] for r in rows])
    f1_span, f1_abstain = per_example_f1(rows)

    source = "given"
    if threshold is None:
        threshold = choose_threshold(scores, f1_span, f1_abstain)
        source = "chosen_on_this_file"

    idx = np.arange(len(rows))
    f1_metric = lambda i, s: float(np.where(s > threshold, f1_span[i], f1_abstain[i]).mean())

    said_present = scores > threshold
    answerable, unanswerable = y == 1, y == 0
    per_category = {}
    by_cat = defaultdict(list)
    for i, r in enumerate(rows):
        by_cat[r["category"]].append(i)
    for cat, ids in sorted(by_cat.items()):
        ids = np.array(ids)
        per_category[cat] = {"n": int(len(ids)), "n_pos": int(y[ids].sum()), "aupr": aupr(y[ids], scores[ids])}
    with_pos = [v["aupr"] for v in per_category.values() if v["aupr"] is not None]

    return {
        "n_examples": len(rows),
        "n_contracts": int(len(np.unique(groups))),
        "prevalence_of_present": float(y.mean()),
        "aupr_chance_level": float(y.mean()),
        "threshold": threshold,
        "threshold_source": source,
        "micro_aupr": ci_dict(bootstrap_ci(y, scores, groups, aupr, n_boot, seed)),
        "micro_precision_at_recall_0.8": ci_dict(bootstrap_ci(y, scores, groups, lambda a, b: precision_at_recall(a, b, 0.8), n_boot, seed)),
        "f1_with_abstention": ci_dict(bootstrap_ci(idx, scores, groups, f1_metric, n_boot, seed)),
        "baseline_always_abstain_f1": float(f1_abstain.mean()),
        "answerable_examples": {
            "n": int(answerable.sum()),
            "said_present_rate": float(said_present[answerable].mean()) if answerable.any() else None,
            "mean_f1": float(np.where(said_present, f1_span, 0.0)[answerable].mean()) if answerable.any() else None,
        },
        "unanswerable_examples": {
            "n": int(unanswerable.sum()),
            "correctly_abstained_rate": float((~said_present)[unanswerable].mean()) if unanswerable.any() else None,
        },
        "macro_aupr": float(np.mean(with_pos)) if with_pos else None,
        "macro_aupr_n_categories": len(with_pos),
        "categories_without_positives": sorted(c for c, v in per_category.items() if v["aupr"] is None),
        "per_category": per_category,
    }


def fmt_ci(d):
    return f"{d['point']:.3f}  (95% CI {d['lo']:.3f} to {d['hi']:.3f})" if d["lo"] is not None else f"{d['point']:.3f}  (CI undefined)"


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()
    guard_threshold(args.pred, args.threshold)

    rows = read_jsonl(args.pred)
    report = compute_report(rows, args.threshold, args.n_boot, cfg.get("seed", 42))
    print(f"examples: {report['n_examples']} | contracts: {report['n_contracts']} | present rate: {report['prevalence_of_present']:.3f}")
    print(f"threshold: {report['threshold']:.3f} ({report['threshold_source']})")
    print(f"micro AUPR:                 {fmt_ci(report['micro_aupr'])}   [chance level {report['aupr_chance_level']:.3f}]")
    print(f"micro precision at 80% recall: {fmt_ci(report['micro_precision_at_recall_0.8'])}")
    print(f"F1 with abstention:         {fmt_ci(report['f1_with_abstention'])}   [always-abstain baseline {report['baseline_always_abstain_f1']:.3f}]")
    a, u = report["answerable_examples"], report["unanswerable_examples"]
    print(f"answerable: n={a['n']} said present {a['said_present_rate']:.3f} mean F1 {a['mean_f1']:.3f} | unanswerable: n={u['n']} correctly abstained {u['correctly_abstained_rate']:.3f}")
    print(f"macro AUPR over {report['macro_aupr_n_categories']} categories with positives: {report['macro_aupr']:.3f}")
    if report["categories_without_positives"]:
        print(f"categories without positives (AUPR undefined): {report['categories_without_positives']}")
    print("\nper category (n, positives, AUPR):")
    for cat, v in sorted(report["per_category"].items(), key=lambda kv: (kv[1]["aupr"] is None, kv[1]["aupr"] or 0)):
        aupr_text = "undefined" if v["aupr"] is None else f"{v['aupr']:.3f}"
        print(f"  {cat:<45} n={v['n']:<4} pos={v['n_pos']:<4} aupr={aupr_text}")

    out = Path(cfg["paths"]["reports_dir"]) / f"eval_{Path(args.pred).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()