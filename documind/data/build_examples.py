"""Build clean examples and contract-level splits from CUAD-QA.

Train rows in CUAD-QA are one row per answer span, so we regroup them into one example
per (contract, category) holding ALL spans. The official test split already works that way.
Validation is carved out of train by contract title.

Run from the project root (E:\\Documind) with the virtual environment active:
    python -m documind.data.build_examples
Writes data\\processed\\ (contracts.json, questions.json, train/val/test.jsonl)
and reports\\split_summary.json
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from datasets import load_dataset

CATEGORY_RE = re.compile(r'related to "(.+?)" that')


def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def group_split(ds, categories):
    """Return (examples, contracts, questions, stats) for one official split."""
    contracts = {}
    questions = {}
    groups = defaultdict(list)       # (title, category) -> list of (start, text)
    row_counts = Counter()           # (title, category) -> rows seen
    empty_rows = Counter()           # (title, category) -> rows with no answer
    for batch in ds.iter(batch_size=128):
        for i in range(len(batch["id"])):
            title = batch["title"][i]
            question = batch["question"][i]
            match = CATEGORY_RE.search(question)
            if not match:
                raise ValueError(f"Cannot parse category from question: {question[:120]!r}")
            category = match.group(1)
            if category not in categories:
                raise ValueError(f"Unknown category {category!r}")
            if category in questions and questions[category] != question:
                raise ValueError(f"Question text differs within category {category!r}")
            questions[category] = question
            if title not in contracts:
                contracts[title] = batch["context"][i]
            key = (title, category)
            row_counts[key] += 1
            texts = batch["answers"][i]["text"]
            starts = batch["answers"][i]["answer_start"]
            if len(texts) == 0:
                empty_rows[key] += 1
            for text, start in zip(texts, starts):
                groups[key].append((int(start), text))

    stats = {
        "pairs_with_empty_and_answer_rows": sum(1 for k in row_counts if empty_rows[k] and groups[k]),
        "pairs_with_several_empty_rows": sum(1 for k in row_counts if empty_rows[k] > 1),
        "duplicate_spans_removed": 0,
        "overlapping_span_pairs": 0,
    }
    examples = []
    for key in sorted(row_counts):
        title, category = key
        spans = sorted(set(groups[key]))
        stats["duplicate_spans_removed"] += len(groups[key]) - len(spans)
        for a, b in zip(spans, spans[1:]):
            if b[0] < a[0] + len(a[1]):
                stats["overlapping_span_pairs"] += 1
        examples.append({
            "title": title,
            "category": category,
            "answers": [{"text": t, "start": s} for s, t in spans],
        })
    return examples, contracts, questions, stats


def summarize(examples, categories):
    n = len(examples)
    empty = sum(1 for e in examples if not e["answers"])
    pos_per_cat = Counter(e["category"] for e in examples if e["answers"])
    return {
        "contracts": len({e["title"] for e in examples}),
        "examples": n,
        "no_answer_examples": empty,
        "no_answer_fraction": round(empty / n, 4) if n else None,
        "answer_spans": sum(len(e["answers"]) for e in examples),
        "min_positives_in_a_category": min(pos_per_cat.get(c, 0) for c in categories),
        "categories_with_fewer_than_5_positives": sorted(c for c in categories if pos_per_cat.get(c, 0) < 5),
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    cfg = load_config()
    categories = cfg["categories"]
    ds = load_dataset(cfg["data"]["hf_dataset"], cache_dir=cfg["data"]["raw_dir"])

    all_contracts, all_questions, split_examples, group_stats = {}, {}, {}, {}
    for name in ("train", "test"):
        print(f"Grouping official split '{name}' ({len(ds[name])} rows) ...")
        examples, contracts, questions, stats = group_split(ds[name], categories)
        split_examples[name] = examples
        group_stats[name] = stats
        for title, text in contracts.items():
            if title in all_contracts:
                raise ValueError(f"Contract {title!r} appears in two official splits")
            all_contracts[title] = text
        for c, q in questions.items():
            if c in all_questions and all_questions[c] != q:
                raise ValueError(f"Question text for {c!r} differs between train and test")
            all_questions[c] = q

    # Every contract must have exactly one example per category.
    for name, examples in split_examples.items():
        per_title = Counter(e["title"] for e in examples)
        bad = {t: n for t, n in per_title.items() if n != len(categories)}
        if bad:
            raise ValueError(f"{name}: {len(bad)} contracts do not have {len(categories)} categories, e.g. {list(bad.items())[:3]}")

    # Contract-level validation split carved from train.
    train_titles = sorted({e["title"] for e in split_examples["train"]})
    rng = random.Random(cfg["seed"])
    rng.shuffle(train_titles)
    n_val = round(len(train_titles) * cfg["data"]["val_fraction_of_train"])
    val_titles = set(train_titles[:n_val])
    final = {
        "train": [e for e in split_examples["train"] if e["title"] not in val_titles],
        "val": [e for e in split_examples["train"] if e["title"] in val_titles],
        "test": split_examples["test"],
    }

    titles = {k: {e["title"] for e in v} for k, v in final.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = titles[a] & titles[b]
        if overlap:
            raise ValueError(f"Leakage: {len(overlap)} contracts shared between {a} and {b}")

    report = {"grouping_stats": group_stats, "splits": {k: summarize(v, categories) for k, v in final.items()}}
    for name, s in report["splits"].items():
        print(f"\n=== {name} ===")
        for key, value in s.items():
            print(f"{key}: {value}")
    print("\n=== grouping checks (train / test) ===")
    for name, s in group_stats.items():
        print(name, s)
    print("\nNo contract appears in more than one split: OK")

    out = Path(cfg["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "contracts.json").write_text(json.dumps(all_contracts, ensure_ascii=False), encoding="utf-8")
    (out / "questions.json").write_text(json.dumps(all_questions, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, rows in final.items():
        write_jsonl(out / f"{name}.jsonl", rows)
    rep_dir = Path(cfg["paths"]["reports_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "split_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}\\ (contracts.json, questions.json, train/val/test.jsonl) and reports\\split_summary.json")


if __name__ == "__main__":
    main()