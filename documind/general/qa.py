"""General mode: ask a question about any PDF. Hybrid retrieval (BM25 + dense) finds the excerpts, a Groq-hosted LLM answers from them.

Network use is opt-in. Only the question and the few retrieved excerpts are sent to Groq, never the whole file.
The API key is read from the GROQ_API_KEY environment variable and is never stored, logged or returned.
"""
import json
import os
import re
import threading
import urllib.error
import urllib.request
from collections import Counter

import numpy as np

from documind.pdf.extract import span_to_locations
from documind.retrieval.bm25 import BM25, tokenize
from documind.retrieval.chunking import chunk_text

KEY_ENV = "GROQ_API_KEY"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
NOT_FOUND = "Not found in the document."
SYSTEM_PROMPT = (
    "You answer questions about a document using only the numbered excerpts you are given. "
    "Cite the excerpts you used right after each claim, like [1] or [2][3]. "
    f"If the excerpts do not contain the answer, reply exactly: {NOT_FOUND} "
    "Never use outside knowledge. Keep the answer short and plain."
)
DEFAULTS = {"model": "openai/gpt-oss-120b", "base_url": "https://api.groq.com/openai/v1", "embed_model": "BAAI/bge-small-en-v1.5",
            "top_k": 6, "chunk_size": 120, "chunk_stride": 60, "timeout_s": 60, "max_question_chars": 500,
            "max_tokens": 1500, "reasoning_effort": "low"}


def load_settings(path=None):
    cfg = dict(DEFAULTS)
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def rrf(orders, k=60):
    """Reciprocal rank fusion of rankings (lists of chunk indices, best first)."""
    score = {}
    for order in orders:
        for rank, idx in enumerate(order, start=1):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda i: (-score[i], i))


class Retriever:
    """BM25 plus dense retrieval over the overlapping word chunks of one document."""

    def __init__(self, doc, size=120, stride=60, embedder=None):
        self.doc = doc
        self.chunks = chunk_text(doc["text"], size, stride)
        if not self.chunks:
            raise ValueError("This document has no text to search.")
        self.tokens = [tokenize(c["text"]) for c in self.chunks]
        self.counts = [Counter(t) for t in self.tokens]
        self.bm25 = BM25().fit(self.tokens)
        self.embedder = embedder
        self.index = None
        if embedder is not None:
            import faiss
            vecs = np.ascontiguousarray(embedder.encode([c["text"] for c in self.chunks]), dtype=np.float32)
            self.index = faiss.IndexFlatIP(vecs.shape[1])
            self.index.add(vecs)

    def search(self, query, k=6):
        q_tokens = tokenize(query)
        scores = [self.bm25.score(q_tokens, c, len(t)) for c, t in zip(self.counts, self.tokens)]
        bm25_order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        orders = [bm25_order]
        if self.index is not None:
            qv = np.ascontiguousarray(self.embedder.encode([QUERY_PREFIX + query]), dtype=np.float32)
            _, ids = self.index.search(qv, self.index.ntotal)
            orders.append([int(i) for i in ids[0]])
        out = []
        for idx in rrf(orders)[:k]:
            c = self.chunks[idx]
            loc = span_to_locations(self.doc, c["char_start"], c["char_end"])
            out.append({"text": c["text"], "char_start": c["char_start"], "char_end": c["char_end"],
                        "pages": loc["pages"], "highlights": loc["highlights"]})
        return out


class SentenceEmbedder:
    """Wraps a sentence-transformers model so that encode() returns unit-length vectors."""

    def __init__(self, name):
        import torch
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device="cuda" if torch.cuda.is_available() else "cpu")

    def encode(self, texts):
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def build_messages(question, passages):
    blocks = []
    for i, p in enumerate(passages, start=1):
        pages = ", ".join(str(n) for n in p["pages"]) or "?"
        blocks.append(f"[{i}] (page {pages}) {p['text']}")
    user = "Excerpts:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def groq_chat(messages, model, base_url, api_key, timeout=60, max_tokens=1500, reasoning_effort=None):
    """One chat completion from a Groq (OpenAI-compatible) endpoint. Raises ValueError with a plain message on any failure."""
    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body, method="POST",
                                 headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json",
                                          "User-Agent": "DocuMind/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ValueError("Groq rejected the API key. Check the GROQ_API_KEY environment variable.")
        if exc.code == 429:
            raise ValueError("Groq says the rate limit was reached. Wait a minute and try again.")
        if exc.code == 404:
            raise ValueError(f"Groq does not know the model {model!r}. Change it in configs/general.json.")
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8", "replace")).get("error", {}).get("message", "")
        except Exception:
            detail = ""
        detail = " ".join(str(detail).split()).replace(api_key, "[hidden]")[:200]
        raise ValueError(f"Groq returned an error (status {exc.code})." + (f" {detail}" if detail else ""))
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ValueError("Could not reach Groq. Check the internet connection.")
    try:
        choice = data["choices"][0]
        answer = (choice["message"].get("content") or "").strip()
        finish = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ValueError("Groq sent a reply in an unexpected format.")
    if not answer:
        if finish == "length":
            raise ValueError("The model used its whole answer budget before replying. Raise max_tokens in configs/general.json and try again.")
        raise ValueError("Groq sent an empty answer. Try asking again.")
    return {"answer": answer, "usage": data.get("usage")}


def parse_answer(answer, n_passages):
    found = not answer.strip().lower().startswith(NOT_FOUND.lower().rstrip("."))
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= n_passages})
    return found, cited


class GeneralMode:
    def __init__(self, settings=None, chat_fn=None, embedder=None, use_dense=True):
        self.cfg = settings or dict(DEFAULTS)
        self.chat_fn = chat_fn            # tests inject a fake; real use calls groq_chat
        self._embedder = embedder
        self.use_dense = use_dense
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, path="configs/general.json"):
        return cls(load_settings(path))

    def available(self):
        return self.chat_fn is not None or bool(os.environ.get(KEY_ENV))

    def info(self):
        return {"available": self.available(), "model": self.cfg["model"], "key_variable": KEY_ENV}

    def _get_embedder(self):
        if not self.use_dense:
            return None
        if self._embedder is None:
            self._embedder = SentenceEmbedder(self.cfg["embed_model"])
        return self._embedder

    def retriever_for(self, rec):
        with self._lock:
            if rec.get("retriever") is None:
                rec["retriever"] = Retriever(rec["doc"], self.cfg["chunk_size"], self.cfg["chunk_stride"], self._get_embedder())
            return rec["retriever"]

    def ask(self, rec, question):
        passages = self.retriever_for(rec).search(question, self.cfg["top_k"])
        messages = build_messages(question, passages)
        if self.chat_fn is not None:
            reply = self.chat_fn(messages)
        else:
            reply = groq_chat(messages, self.cfg["model"], self.cfg["base_url"], os.environ[KEY_ENV], self.cfg["timeout_s"],
                              self.cfg.get("max_tokens", 1500), self.cfg.get("reasoning_effort") or None)
        found, cited = parse_answer(reply["answer"], len(passages))
        return {"question": question, "answer": reply["answer"], "found": found, "cited": cited, "model": self.cfg["model"],
                "passages": passages}
