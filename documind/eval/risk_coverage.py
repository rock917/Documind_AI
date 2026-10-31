"""Risk-coverage analysis of the abstention threshold.

For every threshold t on the "clause is present" score the system outputs a span when score > t.
  coverage = share of ALL (contract, category) cases where a span is output
  risk     = share of those outputs that are wrong
  recall   = share of present clauses that were output correctly
An output is correct only if the clause exists AND the predicted text matches a gold span with token F1 >= 0.5.
Operating points (best coverage under a risk limit) are chosen on the file you pass, so use VAL for that.
On a test file only --threshold is accepted (no tuning on test).

Run from the project root (E:\\Documind) with the virtual environment active:
    python -m documind.eval.risk_coverage --pred data\\predictions\\legalbert_qa_val.jsonl --threshold 8.69
Writes reports\\risk_coverage_<file name>.json and .png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from documind.eval.metrics import best_f1

NEG_INF = -1e9
MATCH_F1 = 0.5


def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def correct_flags(rows, match_f1=MATCH_F1):
    """1 if outputting the predicted span would be correct, else 0 (including every absent clause)."""
    flags = []
    for r in rows:
        golds = [a["text"] for a in r["answers"]]
        flags.append(1 if golds and best_f1(r["pred_text"], golds) >= match_f1 else 0)
    return np.array(flags)


def risk_coverage_curve(scores, correct, n_answerable):
    """Sort by score (high first); at each cutoff k return coverage, risk, recall and the score at the cutoff."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=float)
    order = np.argsort(-scores, kind="stable")
    c = correct[order]
    k = np.arange(1, len(c) + 1)
    hits = np.cumsum(c)
    return {
        "coverage": k / len(c),
        "risk": 1.0 - hits / k,
        "recall": hits / max(n_answerable, 1),
        "score": scores[order],
    }


def area_under_risk_coverage(curve):
    return float(np.mean(curve["risk"]))


def operating_point(curve, max_risk):
    """Largest coverage whose risk is at most max_risk. The threshold is the score at that cutoff (answer if score >= it)."""
    ok = np.flatnonzero(curve["risk"] <= max_risk)
    if len(ok) == 0:
        return None
    i = int(ok[-1])
    return {"max_risk": max_risk, "coverage": float(curve["coverage"][i]), "risk": float(curve["risk"][i]),
            "recall": float(curve["recall"][i]), "answer_if_score_at_least": float(curve["score"][i])}


def point_at_threshold(scores, correct, n_answerable, threshold):
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=float)
    out = scores > threshold
    n_out = int(out.sum())
    return {
        "threshold": threshold,
        "coverage": float(out.mean()),
        "risk": float(1.0 - correct[out].mean()) if n_out else None,
        "recall": float(correct[out].sum() / max(n_answerable, 1)),
        "n_outputs": n_out,
    }


def save_plot(curve, point, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(curve["coverage"], curve["risk"], color="#1f77b4")
    axes[0].set_xlabel("coverage (share of cases where a span is output)")
    axes[0].set_ylabel("risk (share of outputs that are wrong)")
    axes[0].set_title("Risk-coverage")
    axes[1].plot(curve["recall"], 1.0 - curve["risk"], color="#1f77b4")
    axes[1].set_xlabel("recall of present clauses")
    axes[1].set_ylabel("precision of outputs")
    axes[1].set_title("Precision-recall")
    if point and point["risk"] is not None:
        axes[0].scatter([point["coverage"]], [point["risk"]], color="#d62728", zorder=3, label="chosen threshold")
        axes[1].scatter([point["recall"]], [1.0 - point["risk"]], color="#d62728", zorder=3, label="chosen threshold")
        axes[0].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--threshold", type=float, default=None, help="mark this threshold (required for test files)")
    args = p.parse_args()
    is_test = args.pred.lower().endswith("_test.jsonl")
    if is_test and args.threshold is None:
        raise ValueError("For test predictions pass --threshold chosen on val. No operating points are chosen on test.")

    rows = read_jsonl(args.pred)
    scores = np.array([r["present_score"] if r["present_score"] is not None else NEG_INF for r in rows], dtype=float)
    correct = correct_flags(rows)
    n_answerable = int(sum(1 for r in rows if r["answers"]))
    curve = risk_coverage_curve(scores, correct, n_answerable)

    report = {"n_examples": len(rows), "n_answerable": n_answerable, "match_f1": MATCH_F1,
              "area_under_risk_coverage": area_under_risk_coverage(curve)}
    print(f"examples: {len(rows)} | answerable: {n_answerable} | correct = clause present and token F1 >= {MATCH_F1}")
    print(f"area under the risk-coverage curve (lower is better): {report['area_under_risk_coverage']:.4f}")

    point = None
    if args.threshold is not None:
        point = point_at_threshold(scores, correct, n_answerable, args.threshold)
        report["at_threshold"] = point
        risk_text = "n/a" if point["risk"] is None else f"{point['risk']:.3f}"
        print(f"\nat threshold {args.threshold:.3f}: coverage {point['coverage']:.3f} | risk {risk_text} | recall {point['recall']:.3f} | outputs {point['n_outputs']}")

    if not is_test:
        print("\ncoverage -> risk, recall (answering the highest-scoring cases first):")
        table = []
        for target in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00):
            i = max(int(round(target * len(rows))) - 1, 0)
            table.append({"coverage": float(curve["coverage"][i]), "risk": float(curve["risk"][i]),
                          "recall": float(curve["recall"][i]), "score": float(curve["score"][i])})
            print(f"  coverage {table[-1]['coverage']:.2f}  risk {table[-1]['risk']:.3f}  recall {table[-1]['recall']:.3f}  (score >= {table[-1]['score']:.2f})")
        report["table"] = table
        print("\nbest coverage under a risk limit (choose on val only):")
        report["operating_points"] = []
        for max_risk in (0.02, 0.05, 0.10):
            op = operating_point(curve, max_risk)
            report["operating_points"].append(op)
            if op is None:
                print(f"  risk <= {max_risk:.2f}: not reachable")
            else:
                print(f"  risk <= {max_risk:.2f}: coverage {op['coverage']:.3f}, recall {op['recall']:.3f}, answer if score >= {op['answer_if_score_at_least']:.2f}")

    stem = Path(args.pred).stem
    out_dir = Path(cfg["paths"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"risk_coverage_{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_plot(curve, point, out_dir / f"risk_coverage_{stem}.png", stem)
    print(f"\nSaved {out_dir / ('risk_coverage_' + stem + '.json')} and the .png next to it")


if __name__ == "__main__":
    main()