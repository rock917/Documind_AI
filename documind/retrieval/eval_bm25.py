"""Evaluate BM25 passage retrieval: does a chunk containing the gold answer appear in the top k?

Contracts are split into overlapping 120-word chunks. BM25 ranks the chunks of one contract for one category
question; IDF is fitted on the chunks of the TRAIN contracts only. A chunk is correct when it contains a gold
answer (see chunk_contains_answer). Reports Recall@k and MRR with bootstrap CIs over contracts, the exact
random-ranking baseline, and per-category Recall@5.

Run from the project root (E:\\Documind) with the virtual environment active:
    python -m documind.retrieval.eval_bm25 --split val
Writes reports\\bm25_retrieval_<split>.json
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from documind.eval.metrics import bootstrap_ci
from documind.retrieval.bm25 import BM25, tokenize
from documind.retrieval.chunking import chunk_contains_answer, chunk_text

KS = (1, 3, 5, 10, 20)


def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_query(category, question):
    """Category name plus the category description (the text after 'Details:' in the question), if any."""
    details = question.split("Details:", 1)[1].strip() if "Details:" in question else ""
    return f"{category} {details}".strip()


def first_relevant_rank(order, relevant):
    """1-based rank of the first relevant chunk in `order` (chunk indices, best first), or None."""
    for rank, idx in enumerate(order, start=1):
        if relevant[idx]:
            return rank
    return None


def random_recall_at_k(n_chunks, n_relevant, k):
    """Exact chance that a random ranking puts at least one of n_relevant chunks in the top k."""
    if n_relevant <= 0:
        return 0.0
    if n_chunks - n_relevant < k:
        return 1.0
    return 1.0 - math.comb(n_chunks - n_relevant, k) / math.comb(n_chunks, k)


def ci_dict(result):
    point, lo, hi = result
    return {"point": point, "lo": lo, "hi": hi}


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--final", action="store_true", help="required for the test split (the test set is used once, at the end)")
    p.add_argument("--chunk-size", type=int, default=120)
    p.add_argument("--chunk-stride", type=int, default=60)
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()
    if args.split == "test" and not args.final:
        raise ValueError("The test split is kept for the final comparison. Pass --final only when you mean it.")

    processed = Path(cfg["data"]["processed_dir"])
    contracts = json.loads((processed / "contracts.json").read_text(encoding="utf-8"))
    questions = json.loads((processed / "questions.json").read_text(encoding="utf-8"))
    train_titles = sorted({e["title"] for e in read_jsonl(processed / "train.jsonl")})
    examples = [e for e in read_jsonl(processed / f"{args.split}.jsonl") if e["answers"]]

    print(f"fitting BM25 IDF on the chunks of {len(train_titles)} train contracts ...")
    bm = BM25().fit(tokenize(c["text"]) for t in train_titles for c in chunk_text(contracts[t], args.chunk_size, args.chunk_stride))

    cache = {}
    def prepared(title):
        if title not in cache:
            chunks = chunk_text(contracts[title], args.chunk_size, args.chunk_stride)
            toks = [tokenize(c["text"]) for c in chunks]
            cache[title] = (chunks, [Counter(t) for t in toks], [len(t) for t in toks])
        return cache[title]

    ranks, n_chunks_list, n_rel_list, titles, cats = [], [], [], [], []
    no_relevant = 0
    for e in examples:
        chunks, counts, lens = prepared(e["title"])
        query = tokenize(build_query(e["category"], questions[e["category"]]))
        scores = [bm.score(query, c, l) for c, l in zip(counts, lens)]
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])  # stable: ties keep chunk order
        relevant = [chunk_contains_answer(c, e["answers"]) for c in chunks]
        if not any(relevant):
            no_relevant += 1
        ranks.append(first_relevant_rank(order, relevant))
        n_chunks_list.append(len(chunks))
        n_rel_list.append(sum(relevant))
        titles.append(e["title"])
        cats.append(e["category"])

    groups = np.array(titles)
    report = {"split": args.split, "n_present": len(examples), "chunk_size": args.chunk_size, "chunk_stride": args.chunk_stride,
              "mean_chunks_per_contract": float(np.mean([len(prepared(t)[0]) for t in set(titles)])),
              "examples_with_no_relevant_chunk": no_relevant, "recall": {}, "random_recall": {}}
    example_query_cat = examples[0]["category"]
    print(f"example query ({example_query_cat}): {build_query(example_query_cat, questions[example_query_cat])[:300]}")
    print(f"present clauses: {len(examples)} | mean chunks per contract: {report['mean_chunks_per_contract']:.0f} | no relevant chunk at all: {no_relevant}")
    print("\nRecall@k (a chunk containing the gold answer is in the top k)   [random ranking in brackets]")
    for k in KS:
        hit = np.array([1 if (r is not None and r <= k) else 0 for r in ranks])
        res = ci_dict(bootstrap_ci(hit, np.zeros(len(hit)), groups, lambda a, b: float(a.mean()), args.n_boot, cfg.get("seed", 42)))
        rnd = float(np.mean([random_recall_at_k(n, r, k) for n, r in zip(n_chunks_list, n_rel_list)]))
        report["recall"][f"@{k}"] = res
        report["random_recall"][f"@{k}"] = rnd
        print(f"  Recall@{k:<3} {res['point']:.3f} (CI {res['lo']:.3f} to {res['hi']:.3f})   [{rnd:.3f}]")
    mrr = float(np.mean([1.0 / r if r else 0.0 for r in ranks]))
    report["mrr"] = mrr
    print(f"  MRR       {mrr:.3f}")

    by_cat = defaultdict(list)
    for c, r in zip(cats, ranks):
        by_cat[c].append(1 if (r is not None and r <= 5) else 0)
    table = sorted(((c, len(v), float(np.mean(v))) for c, v in by_cat.items() if len(v) >= 5), key=lambda x: x[2])
    report["per_category_recall@5"] = {c: {"n_present": n, "recall@5": r} for c, n, r in table}
    print("\nlowest Recall@5 categories (at least 5 present clauses):")
    for c, n, r in table[:10]:
        print(f"  {c:<42} n={n:<3} {r:.2f}")
    print("highest:")
    for c, n, r in table[-5:]:
        print(f"  {c:<42} n={n:<3} {r:.2f}")

    out = Path(cfg["paths"]["reports_dir"]) / f"bm25_retrieval_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()