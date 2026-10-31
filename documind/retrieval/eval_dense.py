"""Compare BM25, dense (sentence-transformers + FAISS) and hybrid (reciprocal rank fusion) passage retrieval.

Same protocol as eval_bm25: 120-word chunks, one ranking per (contract, category) question, a chunk is correct when it
contains a gold answer. BM25 IDF is fitted on TRAIN contracts only. Dense needs no training: chunks and queries are
embedded with a pretrained model and searched with a FAISS inner-product index (cosine, vectors are normalised).
Hybrid = reciprocal rank fusion (k=60) of the BM25 and dense rankings.

Run from E:\\Documind with the virtual environment active:
    python -m documind.retrieval.eval_dense --split val
Writes reports\\retrieval_compare_<split>.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np

from documind.eval.metrics import bootstrap_ci
from documind.retrieval.bm25 import BM25, tokenize
from documind.retrieval.chunking import chunk_contains_answer, chunk_text
from documind.retrieval.eval_bm25 import KS, build_query, ci_dict, first_relevant_rank, load_config, random_recall_at_k, read_jsonl

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
METHODS = ("bm25", "dense", "hybrid")


def dense_order(index, query_vec):
    """All chunk indices of a FAISS inner-product index, best match first."""
    _, ids = index.search(np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32), index.ntotal)
    return [int(i) for i in ids[0]]


def rrf_order(orders, k=60):
    """Reciprocal rank fusion of several rankings (each a list of chunk indices, best first). Ties go to the lower index."""
    score = {}
    for order in orders:
        for rank, idx in enumerate(order, start=1):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda i: (-score[i], i))


def hit_array(ranks, k):
    return np.array([1 if (r is not None and r <= k) else 0 for r in ranks])


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--final", action="store_true", help="required for the test split (the test set is used once, at the end)")
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--query-prefix", default=None, help="text put before each query; default is the bge instruction for bge models")
    p.add_argument("--chunk-size", type=int, default=120)
    p.add_argument("--chunk-stride", type=int, default=60)
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()
    if args.split == "test" and not args.final:
        raise ValueError("The test split is kept for the final comparison. Pass --final only when you mean it.")
    prefix = args.query_prefix if args.query_prefix is not None else (BGE_QUERY_PREFIX if "bge" in args.model.lower() else "")

    processed = Path(cfg["data"]["processed_dir"])
    contracts = json.loads((processed / "contracts.json").read_text(encoding="utf-8"))
    questions = json.loads((processed / "questions.json").read_text(encoding="utf-8"))
    train_titles = sorted({e["title"] for e in read_jsonl(processed / "train.jsonl")})
    examples = [e for e in read_jsonl(processed / f"{args.split}.jsonl") if e["answers"]]

    print(f"fitting BM25 IDF on the chunks of {len(train_titles)} train contracts ...")
    bm = BM25().fit(tokenize(c["text"]) for t in train_titles for c in chunk_text(contracts[t], args.chunk_size, args.chunk_stride))

    from sentence_transformers import SentenceTransformer
    print(f"loading embedding model {args.model} ...")
    model = SentenceTransformer(args.model)
    print(f"device: {model.device} | query prefix: {prefix!r}")
    cat_names = sorted({e["category"] for e in examples})
    qvecs = model.encode([prefix + build_query(c, questions[c]) for c in cat_names], batch_size=32,
                         normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    qvec = {c: qvecs[i] for i, c in enumerate(cat_names)}

    by_title = defaultdict(list)
    for e in examples:
        by_title[e["title"]].append(e)

    rows = []  # one dict per example
    n_chunks_total = 0
    for n_done, (title, exs) in enumerate(sorted(by_title.items()), start=1):
        chunks = chunk_text(contracts[title], args.chunk_size, args.chunk_stride)
        n_chunks_total += len(chunks)
        toks = [tokenize(c["text"]) for c in chunks]
        counts, lens = [Counter(t) for t in toks], [len(t) for t in toks]
        emb = model.encode([c["text"] for c in chunks], batch_size=32, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        for e in exs:
            query = tokenize(build_query(e["category"], questions[e["category"]]))
            scores = [bm.score(query, c, l) for c, l in zip(counts, lens)]
            order_bm = sorted(range(len(chunks)), key=lambda i: -scores[i])
            order_de = dense_order(index, qvec[e["category"]])
            order_hy = rrf_order([order_bm, order_de])
            relevant = [chunk_contains_answer(c, e["answers"]) for c in chunks]
            rows.append({"title": title, "category": e["category"], "n_chunks": len(chunks), "n_relevant": sum(relevant),
                         "rank": {"bm25": first_relevant_rank(order_bm, relevant),
                                  "dense": first_relevant_rank(order_de, relevant),
                                  "hybrid": first_relevant_rank(order_hy, relevant)}})
        if n_done % 10 == 0 or n_done == len(by_title):
            print(f"  contracts done: {n_done}/{len(by_title)}")

    groups = np.array([r["title"] for r in rows])
    seed = cfg.get("seed", 42)
    ranks = {m: [r["rank"][m] for r in rows] for m in METHODS}
    mean_fn = lambda a, b: float(a.mean())
    report = {"split": args.split, "model": args.model, "query_prefix": prefix, "n_present": len(rows),
              "mean_chunks_per_contract": n_chunks_total / len(by_title), "recall": {m: {} for m in METHODS},
              "random_recall": {}, "mrr": {}, "paired_diff": {}, "per_category_recall@5": {}}

    print(f"\npresent clauses: {len(rows)} | mean chunks per contract: {report['mean_chunks_per_contract']:.0f}")
    print("\nRecall@k  (point estimate; 95% CI over contracts in brackets)")
    print(f"  {'':<10}{'random':>8}  {'BM25':>22}  {'dense':>22}  {'hybrid':>22}")
    for k in KS:
        rnd = float(np.mean([random_recall_at_k(r["n_chunks"], r["n_relevant"], k) for r in rows]))
        report["random_recall"][f"@{k}"] = rnd
        cells = []
        for m in METHODS:
            res = ci_dict(bootstrap_ci(hit_array(ranks[m], k), np.zeros(len(rows)), groups, mean_fn, args.n_boot, seed))
            report["recall"][m][f"@{k}"] = res
            cells.append(f"{res['point']:.3f} [{res['lo']:.3f}-{res['hi']:.3f}]")
        print(f"  Recall@{k:<4}{rnd:>8.3f}  {cells[0]:>22}  {cells[1]:>22}  {cells[2]:>22}")
    for m in METHODS:
        report["mrr"][m] = float(np.mean([1.0 / r if r else 0.0 for r in ranks[m]]))
    print(f"  {'MRR':<10}{'':>8}  {report['mrr']['bm25']:>22.3f}  {report['mrr']['dense']:>22.3f}  {report['mrr']['hybrid']:>22.3f}")

    print("\npaired difference in Recall@k (same clauses, resampling contracts):")
    for a, b in (("dense", "bm25"), ("hybrid", "bm25"), ("hybrid", "dense")):
        for k in (5, 10):
            diff = hit_array(ranks[a], k) - hit_array(ranks[b], k)
            res = ci_dict(bootstrap_ci(diff, np.zeros(len(rows)), groups, mean_fn, args.n_boot, seed))
            report["paired_diff"][f"{a}_minus_{b}@{k}"] = res
            verdict = "clear gain" if res["lo"] > 0 else ("clear loss" if res["hi"] < 0 else "not clearly different")
            print(f"  {a} - {b} @{k:<3} {res['point']:+.3f} (CI {res['lo']:+.3f} to {res['hi']:+.3f})  {verdict}")

    cat_hits = defaultdict(lambda: {m: [] for m in METHODS})
    for r in rows:
        for m in METHODS:
            rk = r["rank"][m]
            cat_hits[r["category"]][m].append(1 if (rk is not None and rk <= 5) else 0)
    table = []
    for c, d in cat_hits.items():
        n = len(d["bm25"])
        if n >= 5:
            table.append((c, n, {m: float(np.mean(d[m])) for m in METHODS}))
            report["per_category_recall@5"][c] = {"n_present": n, **{m: float(np.mean(d[m])) for m in METHODS}}
    table.sort(key=lambda t: t[2]["dense"] - t[2]["bm25"])
    print("\nRecall@5 per category where dense is most BEHIND BM25, then most AHEAD (categories with 5+ present clauses):")
    print(f"  {'category':<42}{'n':>4}{'BM25':>7}{'dense':>7}{'hybrid':>8}")
    shown = table if len(table) <= 10 else table[:5] + table[-5:]
    for i, (c, n, v) in enumerate(shown):
        if len(table) > 10 and i == 5:
            print("  ...")
        print(f"  {c:<42}{n:>4}{v['bm25']:>7.2f}{v['dense']:>7.2f}{v['hybrid']:>8.2f}")

    report["examples"] = [{"title": r["title"], "category": r["category"], **{f"rank_{m}": r["rank"][m] for m in METHODS}} for r in rows]
    out = Path(cfg["paths"]["reports_dir"]) / f"retrieval_compare_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()