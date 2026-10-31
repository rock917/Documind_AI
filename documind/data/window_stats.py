"""Measure sliding-window statistics for CUAD with the real tokenizer (no training).

Answers three questions before we build training data:
  1. How many windows does a contract produce at the chosen max_length and stride?
  2. What share of windows contain an answer (the rest are negatives we must subsample)?
  3. How many gold answers do NOT fit fully inside any single window?

Run from the project root (E:\\Documind) with the virtual environment active:
    python -m documind.data.window_stats --split train
Writes reports\\window_stats_<split>.json
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoTokenizer


def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def encode(tok, question, context, max_length, stride):
    return tok(
        question,
        context,
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )


def context_char_ranges(enc):
    """For each window, the (start_char, end_char) span of context text it covers."""
    ranges = []
    for i in range(len(enc["input_ids"])):
        seq = enc.sequence_ids(i)
        offsets = enc["offset_mapping"][i]
        ctx = [offsets[j] for j, s in enumerate(seq) if s == 1]
        ranges.append((ctx[0][0], ctx[-1][1]) if ctx else (0, 0))
    return ranges


def stats(values):
    a = np.asarray(values, dtype=float)
    return {
        "mean": round(float(a.mean()), 1),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--tokenizer", default="nlpaueb/legal-bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-positive-examples", type=int, default=1500,
                        help="random sample of answered examples for the coverage check (keeps it fast)")
    args = parser.parse_args()

    processed = Path(cfg["data"]["processed_dir"])
    contracts = json.loads((processed / "contracts.json").read_text(encoding="utf-8"))
    questions = json.loads((processed / "questions.json").read_text(encoding="utf-8"))
    examples = read_jsonl(processed / f"{args.split}.jsonl")
    n_categories = len(cfg["categories"])

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer: {args.tokenizer} | max_length={args.max_length} stride={args.stride} | split={args.split}")

    # 1. Windows per contract (one representative question; real questions differ by a few tokens).
    titles = sorted({e["title"] for e in examples})
    probe_question = questions[cfg["categories"][0]]
    windows_per_contract = []
    for title in titles:
        enc = encode(tok, probe_question, contracts[title], args.max_length, args.stride)
        windows_per_contract.append(len(enc["input_ids"]))
    est_total = int(sum(windows_per_contract) * n_categories)
    print(f"contracts: {len(titles)} | windows per contract: {stats(windows_per_contract)}")
    print(f"estimated total windows for this split ({n_categories} questions per contract): {est_total}")

    # 2 and 3. Answer coverage on a random sample of answered examples.
    positives = [e for e in examples if e["answers"]]
    rng = random.Random(cfg["seed"])
    if len(positives) > args.max_positive_examples:
        positives = rng.sample(positives, args.max_positive_examples)
    n_answers = fit_answers = 0
    n_windows = pos_windows = 0
    unfit_lengths = []
    for e in positives:
        enc = encode(tok, questions[e["category"]], contracts[e["title"]], args.max_length, args.stride)
        ranges = context_char_ranges(enc)
        n_windows += len(ranges)
        window_has_answer = [False] * len(ranges)
        for a in e["answers"]:
            n_answers += 1
            s, t = a["start"], a["start"] + len(a["text"])
            fits = False
            for w, (ws, we) in enumerate(ranges):
                if ws <= s and t <= we:
                    fits = True
                    window_has_answer[w] = True
            fit_answers += fits
            if not fits:
                unfit_lengths.append(len(a["text"].split()))
        pos_windows += sum(window_has_answer)

    result = {
        "split": args.split,
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "stride": args.stride,
        "contracts": len(titles),
        "windows_per_contract": stats(windows_per_contract),
        "estimated_total_windows": est_total,
        "sampled_answered_examples": len(positives),
        "answers_checked": n_answers,
        "answers_fully_inside_some_window": fit_answers,
        "answers_in_no_single_window": n_answers - fit_answers,
        "fraction_answers_in_no_single_window": round((n_answers - fit_answers) / n_answers, 4),
        "windows_in_sample": n_windows,
        "windows_containing_an_answer": pos_windows,
        "fraction_windows_with_answer_in_answered_examples": round(pos_windows / n_windows, 4),
        "unfit_answer_words": stats(unfit_lengths) if unfit_lengths else None,
    }
    print("\n=== answer coverage (answered examples only) ===")
    for key in list(result)[7:]:
        print(f"{key}: {result[key]}")

    out = Path(cfg["paths"]["reports_dir"]) / f"window_stats_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()