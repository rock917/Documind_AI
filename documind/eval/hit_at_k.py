"""hit@k: how often is a correct passage among the top k candidates?

For every PRESENT clause (gold answer exists), a candidate is correct if its token F1 against the best-matching
gold span is >= 0.5 (same rule as the other reports). hit@k = some candidate among the top k is correct.
Reported for all present clauses, and for the subset where the system says "present" (present_score > threshold),
with 95% bootstrap CIs that resample whole contracts, plus per-category hit rates.
Needs a predictions file that contains candidates (predict.py --top-k).

Run from the project root (E:\\Documind) with the virtual environment active:
    python -m documind.eval.hit_at_k --pred data\\predictions\\legalbert_qa_k5_val.jsonl --threshold 8.69
Writes reports\\hit_at_k_<file name>.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from documind.eval.metrics import best_f1, bootstrap_ci

MATCH_F1 = 0.5
KS = (1, 3, 5)


def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def first_correct_rank(row, match_f1=MATCH_F1):
    """1-based rank of the first correct candidate, or None if no candidate is correct."""
    if "candidates" not in row:
        raise ValueError("This predictions file has no candidates. Re-run predict.py (it saves them with --top-k).")
    golds = [a["text"] for a in row["answers"]]
    for rank, cand in enumerate(row["candidates"], start=1):
        if best_f1(cand["text"], golds) >= match_f1:
            return rank
    return None


def hit_flags(rows, ks=KS):
    """For each row, {k: 0/1}: whether a correct candidate is within the top k."""
    out = []
    for r in rows:
        rank = first_correct_rank(r)
        out.append({k: int(rank is not None and rank <= k) for k in ks})
    return out


def ci_dict(result):
    point, lo, hi = result
    return {"point": point, "lo": lo, "hi": hi}


def hit_report(rows, ks=KS, n_boot=1000, seed=42):
    """hit@k with bootstrap CIs over contracts for the given (present) rows."""
    flags = hit_flags(rows, ks)
    groups = np.array([r["title"] for r in rows])
    out = {"n": len(rows)}
    for k in ks:
        y = np.array([f[k] for f in flags])
        out[f"hit@{k}"] = ci_dict(bootstrap_ci(y, np.zeros(len(y)), groups, lambda a, b: float(a.mean()), n_boot, seed))
    return out


def per_category(rows, ks=KS):
    flags = hit_flags(rows, ks)
    by_cat = defaultdict(list)
    for r, f in zip(rows, flags):
        by_cat[r["category"]].append(f)
    return {cat: {"n_present": len(fs), **{f"hit@{k}": float(np.mean([f[k] for f in fs])) for k in ks}}
            for cat, fs in by_cat.items()}


def fmt(d):
    return f"{d['point']:.3f} (CI {d['lo']:.3f} to {d['hi']:.3f})" if d["lo"] is not None else f"{d['point']:.3f}"


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--threshold", type=float, required=True, help="present_score threshold chosen on val")
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()

    rows = read_jsonl(args.pred)
    present = [r for r in rows if r["answers"]]
    flagged = [r for r in present if r["present_score"] is not None and r["present_score"] > args.threshold]

    seed = cfg.get("seed", 42)
    report = {
        "threshold": args.threshold,
        "all_present": hit_report(present, KS, args.n_boot, seed),
        "present_and_flagged": hit_report(flagged, KS, args.n_boot, seed),
        "per_category": per_category(present),
    }
    print(f"present clauses: {len(present)} | of those flagged present at threshold {args.threshold}: {len(flagged)}")
    for label, key in (("ALL present clauses", "all_present"), ("present clauses the system flags as present", "present_and_flagged")):
        r = report[key]
        print(f"\n{label} (n={r['n']}):")
        for k in KS:
            print(f"  hit@{k}: {fmt(r[f'hit@{k}'])}")
    gain = report["all_present"]["hit@5"]["point"] - report["all_present"]["hit@1"]["point"]
    print(f"\ngain from showing 5 passages instead of 1 (all present): {gain:+.3f}")

    cats = sorted(((c, v) for c, v in report["per_category"].items() if v["n_present"] >= 5), key=lambda kv: kv[1]["hit@1"])
    print("\nlowest hit@1 categories (at least 5 present clauses):  n | hit@1 | hit@3 | hit@5")
    for c, v in cats[:12]:
        print(f"  {c:<42} {v['n_present']:<3} {v['hit@1']:.2f}  {v['hit@3']:.2f}  {v['hit@5']:.2f}")

    out = Path(cfg["paths"]["reports_dir"]) / f"hit_at_k_{Path(args.pred).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()