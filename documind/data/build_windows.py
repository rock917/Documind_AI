"""Build sampled training windows (JSONL) for one split with the real tokenizer.

Keeps EVERY positive window, drops partial windows, and keeps only a random fraction of
negative windows: p_hard for negatives from examples that do have an answer (the clause
exists elsewhere in the contract), p_easy for negatives from examples with no answer at all.
Also checks on real data that the labelled token span decodes back to the gold answer.

Run from the project root (E:\\Documind) with the virtual environment active:
    python -m documind.data.build_windows --split val
Writes data\\processed\\windows_<split>.jsonl and reports\\windows_<split>_summary.json
Use this for train and val only. Test windows are never sampled.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

from documind.data.paths import windows_path
from documind.data.windows import NEGATIVE, PARTIAL, POSITIVE, make_windows
from documind.eval.metrics import best_f1


def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def select_windows(windows, rng, has_answer, p_hard, p_easy):
    """Keep all positives, drop partials, keep each negative with probability p_hard / p_easy."""
    p_negative = p_hard if has_answer else p_easy
    kept = []
    for w in windows:
        if w["label"] == POSITIVE:
            kept.append(w)
        elif w["label"] == NEGATIVE and rng.random() < p_negative:
            kept.append(w)
    return kept


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--tokenizer", default="nlpaueb/legal-bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--p-hard", type=float, default=0.10)
    parser.add_argument("--p-easy", type=float, default=0.07)
    parser.add_argument("--tag", default="", help="suffix for the output file names, e.g. _bert for a second tokenizer")
    parser.add_argument("--limit-examples", type=int, default=0, help="random subset for a quick trial (0 = all)")
    args = parser.parse_args()

    processed = Path(cfg["data"]["processed_dir"])
    contracts = json.loads((processed / "contracts.json").read_text(encoding="utf-8"))
    questions = json.loads((processed / "questions.json").read_text(encoding="utf-8"))
    examples = read_jsonl(processed / f"{args.split}.jsonl")
    rng = random.Random(cfg["seed"])
    if args.limit_examples and args.limit_examples < len(examples):
        examples = rng.sample(examples, args.limit_examples)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    out_path = windows_path(processed, args.split, args.tag)
    total, kept = Counter(), Counter()
    f1_values = []
    answered_without_positive = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for e in tqdm(examples, desc=f"windows[{args.split}]"):
            ctx = contracts[e["title"]]
            has_answer = bool(e["answers"])
            wins = make_windows(tok, questions[e["category"]], ctx, e["answers"], args.max_length, args.stride)
            for w in wins:
                total[w["label"]] += 1
            if has_answer and not any(w["label"] == POSITIVE for w in wins):
                answered_without_positive += 1
            golds = [a["text"] for a in e["answers"]]
            for w in select_windows(wins, rng, has_answer, args.p_hard, args.p_easy):
                kept[w["label"]] += 1
                if w["label"] == POSITIVE:
                    s, t = w["start_position"], w["end_position"]
                    decoded = ctx[w["offset_mapping"][s][0]:w["offset_mapping"][t][1]]
                    f1_values.append(best_f1(decoded, golds))
                out.write(json.dumps({
                    "title": e["title"],
                    "category": e["category"],
                    "window_index": w["window_index"],
                    "label": w["label"],
                    "input_ids": w["input_ids"],
                    "token_type_ids": w["token_type_ids"],
                    "start_position": w["start_position"],
                    "end_position": w["end_position"],
                }) + "\n")

    summary = {
        "split": args.split,
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "stride": args.stride,
        "p_hard": args.p_hard,
        "p_easy": args.p_easy,
        "examples_processed": len(examples),
        "windows_before_sampling": dict(total),
        "windows_kept": dict(kept),
        "answered_examples_with_no_positive_window": answered_without_positive,
        "positive_windows_checked_for_roundtrip": len(f1_values),
        "roundtrip_mean_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else None,
        "roundtrip_windows_with_f1_below_1": sum(1 for v in f1_values if v < 1.0),
    }
    print("\n=== summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    rep = Path(cfg["paths"]["reports_dir"]) / f"windows_{args.split}{args.tag}_summary.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path} and {rep}")


if __name__ == "__main__":
    main()