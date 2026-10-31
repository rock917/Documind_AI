"""Error analysis of saved predictions: what goes wrong, per category and per error type.

Present clauses are classified by what the reader did (independent of the threshold):
  hit               predicted text matches a gold span (token F1 >= 0.5)
  wrong_place       predicted span does not overlap any gold span
  overlap_too_short predicted span overlaps a gold span but is shorter and F1 < 0.5
  overlap_too_long  predicted span overlaps a gold span but is not shorter and F1 < 0.5
  no_prediction     no span at all
Absent clauses with present_score > threshold are false alarms.
Run on VAL only: this script refuses test files, because fixing errors seen on test would spoil the final check.

Run from the project root (E:\\Documind) with the virtual environment active:
    python -m documind.eval.error_analysis --pred data\\predictions\\legalbert_qa_val.jsonl --threshold 8.69
Writes reports\\error_analysis_<file name>.json
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from documind.eval.metrics import best_f1

MATCH_F1 = 0.5
NEG_INF = -1e9
TYPES = ["hit", "wrong_place", "overlap_too_short", "overlap_too_long", "no_prediction"]


def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def guard_not_test(pred_path):
    if str(pred_path).lower().endswith("_test.jsonl"):
        raise ValueError("Error analysis is refused on test predictions: use val, so the test set stays a clean final check.")


def classify_present(row, match_f1=MATCH_F1):
    """Error type for a row whose clause is present."""
    if row["char_start"] is None:
        return "no_prediction"
    golds = [a["text"] for a in row["answers"]]
    if best_f1(row["pred_text"], golds) >= match_f1:
        return "hit"
    ps, pe = row["char_start"], row["char_end"]
    best = None
    for a in row["answers"]:
        gs, ge = a["start"], a["start"] + len(a["text"])
        overlap = min(pe, ge) - max(ps, gs)
        if best is None or overlap > best[0]:
            best = (overlap, gs, ge)
    overlap, gs, ge = best
    if overlap <= 0:
        return "wrong_place"
    return "overlap_too_short" if (pe - ps) < (ge - gs) else "overlap_too_long"


def shorten(text, n=200):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "..."


def analyse(rows, threshold, seed=42, samples_per_type=3):
    scores = np.array([r["present_score"] if r["present_score"] is not None else NEG_INF for r in rows], dtype=float)
    present = [i for i, r in enumerate(rows) if r["answers"]]
    absent = [i for i, r in enumerate(rows) if not r["answers"]]

    types = {i: classify_present(rows[i]) for i in present}
    counts = Counter(types.values())
    by_cat = defaultdict(Counter)
    for i, t in types.items():
        by_cat[rows[i]["category"]][t] += 1
    per_category = {}
    for cat, c in by_cat.items():
        n = sum(c.values())
        per_category[cat] = {"n_present": n, "hit_rate": c["hit"] / n,
                             "wrong_place_rate": c["wrong_place"] / n,
                             "overlap_rate": (c["overlap_too_short"] + c["overlap_too_long"]) / n}

    false_alarms = [i for i in absent if scores[i] > threshold]
    fa_by_cat = Counter(rows[i]["category"] for i in false_alarms)

    def words(text):
        return len(text.split())
    pred_len = [words(rows[i]["pred_text"]) for i in present if rows[i]["char_start"] is not None]
    gold_len = [min(words(a["text"]) for a in rows[i]["answers"]) for i in present]
    rng = random.Random(seed)
    samples = {}
    for t in TYPES[1:4]:
        ids = [i for i in present if types[i] == t]
        samples[t] = [{"category": rows[i]["category"], "score": round(float(scores[i]), 2),
                       "gold": shorten(rows[i]["answers"][0]["text"]), "predicted": shorten(rows[i]["pred_text"])}
                      for i in rng.sample(ids, min(samples_per_type, len(ids)))]
    fa_ids = rng.sample(false_alarms, min(samples_per_type, len(false_alarms)))
    samples["false_alarm"] = [{"category": rows[i]["category"], "score": round(float(scores[i]), 2),
                               "predicted": shorten(rows[i]["pred_text"])} for i in fa_ids]

    return {
        "n_present": len(present),
        "n_absent": len(absent),
        "threshold": threshold,
        "present_error_types": {t: counts[t] for t in TYPES},
        "false_alarms_at_threshold": len(false_alarms),
        "false_alarm_rate_among_absent": len(false_alarms) / max(len(absent), 1),
        "false_alarms_by_category": dict(fa_by_cat.most_common()),
        "per_category": per_category,
        "median_pred_words": float(np.median(pred_len)) if pred_len else None,
        "median_shortest_gold_words": float(np.median(gold_len)) if gold_len else None,
        "samples": samples,
    }


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--threshold", type=float, required=True)
    args = p.parse_args()
    guard_not_test(args.pred)

    rows = read_jsonl(args.pred)
    rep = analyse(rows, args.threshold, cfg.get("seed", 42))
    n = rep["n_present"]
    print(f"present clauses: {n} | absent: {rep['n_absent']} | threshold {args.threshold}")
    print("\nwhat the reader did on PRESENT clauses (independent of the threshold):")
    for t in TYPES:
        c = rep["present_error_types"][t]
        print(f"  {t:<18} {c:>5}  {c / n:6.1%}")
    print(f"\nfalse alarms on ABSENT clauses at the threshold: {rep['false_alarms_at_threshold']} "
          f"({rep['false_alarm_rate_among_absent']:.1%} of absent)")
    print("  by category:", dict(list(rep["false_alarms_by_category"].items())[:10]))
    print(f"\nmedian predicted words: {rep['median_pred_words']} | median gold words (shortest gold per example): {rep['median_shortest_gold_words']}")
    print("\nlowest hit rates (categories with at least 5 present clauses):")
    rows_sorted = sorted(((c, v) for c, v in rep["per_category"].items() if v["n_present"] >= 5), key=lambda kv: kv[1]["hit_rate"])
    for c, v in rows_sorted[:12]:
        print(f"  {c:<42} n={v['n_present']:<3} hit {v['hit_rate']:.2f}  wrong_place {v['wrong_place_rate']:.2f}  overlap {v['overlap_rate']:.2f}")
    print("\nexamples:")
    for t, items in rep["samples"].items():
        print(f"\n[{t}]")
        for s in items:
            print(f"  category={s['category']} score={s['score']}")
            if "gold" in s:
                print(f"    gold:      {s['gold']}")
            print(f"    predicted: {s['predicted']}")

    out = Path(cfg["paths"]["reports_dir"]) / f"error_analysis_{Path(args.pred).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()