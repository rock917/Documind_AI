"""Inspect CUAD-QA before making any design decision.

Reports, per split and combined: rows, contracts, rows per contract, empty answers,
answers per row, answer offsets, context lengths, categories, and leakage between splits.

Run from the project root (E:\\Documind) with the virtual environment active:
    python -m documind.data.inspect_cuad
Writes reports\\cuad_inspection.json
"""
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset

# Question text looks like: Highlight the parts (if any) of this contract related to "Governing Law" that ...
CATEGORY_RE = re.compile(r'related to "(.+?)" that')
ID_SUFFIX_RE = re.compile(r"_(\d+)$")


def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def percentiles(values, qs=(50, 90, 99)):
    """Return [p50, p90, p99, max], or None for an empty list."""
    if len(values) == 0:
        return None
    return [round(float(x), 1) for x in np.percentile(values, qs)] + [float(max(values))]


def inspect_split(ds):
    ctx_info = {}            # title -> (number of characters, hash of the context text)
    ctx_words = {}           # title -> number of words
    ctx_conflicts = set()    # titles that appear with more than one different context
    rows_per_title = Counter()
    cat_rows = Counter()
    cat_empty = Counter()
    answers_per_row = Counter()
    answer_words = []
    id_suffixes = Counter()
    pair_counts = Counter()  # (title, category) -> number of rows
    parse_fail = 0
    id_mismatch = 0
    offsets_checked = 0
    offsets_bad = 0
    n_rows = 0

    for batch in ds.iter(batch_size=128):
        for i in range(len(batch["id"])):
            n_rows += 1
            row_id = batch["id"][i]
            title = batch["title"][i]
            context = batch["context"][i]
            question = batch["question"][i]
            texts = batch["answers"][i]["text"]
            starts = batch["answers"][i]["answer_start"]

            rows_per_title[title] += 1

            match = CATEGORY_RE.search(question)
            category = match.group(1) if match else "UNPARSED"
            if not match:
                parse_fail += 1

            id_tail = row_id.rsplit("__", 1)[-1]
            suffix = ID_SUFFIX_RE.search(id_tail)
            id_suffixes[suffix.group(1) if suffix else "none"] += 1
            if ID_SUFFIX_RE.sub("", id_tail) != category:
                id_mismatch += 1
            pair_counts[(title, category)] += 1

            cat_rows[category] += 1
            if len(texts) == 0:
                cat_empty[category] += 1
            answers_per_row[len(texts)] += 1

            for text, start in zip(texts, starts):
                offsets_checked += 1
                if context[start:start + len(text)] != text:
                    offsets_bad += 1
                answer_words.append(len(text.split()))

            info = (len(context), hash(context))
            if title not in ctx_info:
                ctx_info[title] = info
                ctx_words[title] = len(context.split())
            elif ctx_info[title] != info:
                ctx_conflicts.add(title)

    no_answer_rows = sum(cat_empty.values())
    summary = {
        "rows": n_rows,
        "contracts": len(ctx_info),
        "rows_per_contract_(rows, contracts)": Counter(rows_per_title.values()).most_common(5),
        "rows_with_no_answer": no_answer_rows,
        "no_answer_fraction": round(no_answer_rows / n_rows, 4),
        "answers_per_row": dict(sorted(answers_per_row.items())),
        "offset_checks": offsets_checked,
        "offset_mismatches": offsets_bad,
        "contracts_with_conflicting_context": len(ctx_conflicts),
        "categories": len(cat_rows),
        "question_parse_failures": parse_fail,
        "id_vs_question_category_mismatches": id_mismatch,
        "duplicate_title_category_pairs": sum(1 for c in pair_counts.values() if c > 1),
        "id_suffix_values_top10": dict(sorted(id_suffixes.items(), key=lambda kv: -kv[1])[:10]),
        "answer_words_p50_p90_p99_max": percentiles(answer_words),
        "context_words_p50_p90_p99_max": percentiles(list(ctx_words.values())),
        "context_chars_p50_p90_p99_max": percentiles([c for c, _ in ctx_info.values()]),
    }
    extra = {
        "titles": set(ctx_info),
        "context_hashes": {h for _, h in ctx_info.values()},
        "cat_rows": cat_rows,
        "cat_empty": cat_empty,
    }
    return summary, extra


def main():
    cfg = load_config()
    ds = load_dataset(cfg["data"]["hf_dataset"], cache_dir=cfg["data"]["raw_dir"])

    report = {"splits": {}}
    extras = {}
    for split_name, split in ds.items():
        print(f"Inspecting split '{split_name}' ({len(split)} rows) ...")
        summary, extra = inspect_split(split)
        report["splits"][split_name] = summary
        extras[split_name] = extra
        print(f"\n=== {split_name} ===")
        for key, value in summary.items():
            print(f"{key}: {value}")

    report["leakage"] = {}
    for a, b in combinations(extras, 2):
        shared_titles = len(extras[a]["titles"] & extras[b]["titles"])
        shared_texts = len(extras[a]["context_hashes"] & extras[b]["context_hashes"])
        report["leakage"][f"{a}_vs_{b}"] = {
            "shared_contract_titles": shared_titles,
            "shared_identical_contract_texts": shared_texts,
        }
        print(f"\n=== leakage check: {a} vs {b} ===")
        print(f"shared contract titles: {shared_titles}")
        print(f"shared identical contract texts: {shared_texts}")

    cat_rows, cat_empty = Counter(), Counter()
    for extra in extras.values():
        cat_rows.update(extra["cat_rows"])
        cat_empty.update(extra["cat_empty"])
    table = sorted(
        ((c, cat_rows[c], cat_empty[c], cat_empty[c] / cat_rows[c]) for c in cat_rows),
        key=lambda r: r[3],
    )
    print("\n=== categories (all splits), sorted by fraction of rows with no answer ===")
    for category, rows, empty, fraction in table:
        print(f"{category:<45} rows={rows:<6} no_answer={fraction:.2f}")
    report["categories"] = [
        {"category": c, "rows": n, "no_answer_rows": e, "no_answer_fraction": round(f, 4)}
        for c, n, e, f in table
    ]

    out_dir = Path(cfg["paths"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cuad_inspection.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()