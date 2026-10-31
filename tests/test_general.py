import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from documind.general.qa import (GeneralMode, NOT_FOUND, Retriever, build_messages, groq_chat, parse_answer, rrf)
from documind.pdf.extract import build_document


def doc_from_pages(pages):
    """A document whose pages hold the given texts (one fake box per word)."""
    pages_words = []
    for text in pages:
        pages_words.append([(i * 6.0, (i // 10) * 12.0, i * 6.0 + 5, (i // 10) * 12.0 + 10, w, 0, i // 10, i % 10)
                            for i, w in enumerate(text.split())])
    return build_document(pages_words, [[600.0, 800.0]] * len(pages))


FILLER_1 = " ".join(f"alpha{i % 17} beta{i % 13}" for i in range(250))
FILLER_3 = " ".join(f"gamma{i % 19} delta{i % 11}" for i in range(250))
KEY = "This Agreement shall be governed by the laws of Delaware and disputes go to its courts."
DOC = doc_from_pages([FILLER_1, FILLER_1 + " " + KEY + " " + FILLER_3, FILLER_3])


class HashEmbedder:
    """Bag-of-words vectors from hashes: a stand-in for the real embedding model."""
    def encode(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in t.lower().split():
                out[r, int(hashlib.md5(w.strip(".,").encode()).hexdigest(), 16) % 64] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def test_rrf_favours_items_ranked_well_by_both_lists():
    assert rrf([[0, 1, 2], [2, 0, 1]]) == [0, 2, 1]


def test_bm25_only_retrieval_finds_the_right_page_with_highlights():
    hits = Retriever(DOC).search("which law governs the agreement, Delaware?", k=3)
    assert "Delaware" in hits[0]["text"] and hits[0]["pages"] == [2]
    assert hits[0]["highlights"] and hits[0]["highlights"][0]["page"] == 2


def test_hybrid_retrieval_with_dense_index():
    hits = Retriever(DOC, embedder=HashEmbedder()).search("governed by the laws of Delaware", k=3)
    assert "Delaware" in hits[0]["text"]
    assert len(hits) == 3


def test_empty_document_is_rejected():
    with pytest.raises(ValueError):
        Retriever(build_document([[]], [[600.0, 800.0]]))


def test_messages_number_the_excerpts_and_forbid_outside_knowledge():
    msgs = build_messages("Who?", [{"text": "Acme grants a licence.", "pages": [3]}, {"text": "Beta pays fees.", "pages": []}])
    assert msgs[0]["role"] == "system" and "only" in msgs[0]["content"] and NOT_FOUND in msgs[0]["content"]
    assert "[1] (page 3) Acme grants a licence." in msgs[1]["content"] and "[2] (page ?) Beta pays fees." in msgs[1]["content"]
    assert msgs[1]["content"].endswith("Question: Who?")


def test_parse_answer():
    assert parse_answer("Delaware law applies [1][3] and [9].", 3) == (True, [1, 3])
    assert parse_answer("Not found in the document.", 3) == (False, [])
    assert parse_answer("  not found in the document", 3)[0] is False


class FakeGroq(BaseHTTPRequestHandler):
    status = 200
    seen = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeGroq.seen = {"auth": self.headers["Authorization"], "ua": self.headers["User-Agent"], "path": self.path, "body": body}
        payload = json.dumps({"choices": [{"message": {"content": "  Delaware [1]  "}}], "usage": {"total_tokens": 42}}).encode()
        self.send_response(FakeGroq.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload if FakeGroq.status == 200 else b"{}")

    def log_message(self, *args):
        pass


@pytest.fixture
def groq_url():
    server = HTTPServer(("127.0.0.1", 0), FakeGroq)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    FakeGroq.status = 200
    yield f"http://127.0.0.1:{server.server_port}/openai/v1"
    server.shutdown()


def test_groq_chat_sends_the_expected_request_and_parses_the_reply(groq_url):
    out = groq_chat([{"role": "user", "content": "hi"}], "some-model", groq_url, "secret-key-123")
    assert out == {"answer": "Delaware [1]", "usage": {"total_tokens": 42}}
    seen = FakeGroq.seen
    assert seen["path"] == "/openai/v1/chat/completions" and seen["auth"] == "Bearer secret-key-123"
    assert seen["ua"].startswith("DocuMind") and seen["body"]["model"] == "some-model" and seen["body"]["temperature"] == 0


@pytest.mark.parametrize("status,words", [(401, "rejected"), (429, "rate limit"), (404, "model"), (500, "status 500")])
def test_groq_errors_become_plain_messages_without_the_key(groq_url, status, words):
    FakeGroq.status = status
    with pytest.raises(ValueError) as err:
        groq_chat([], "m", groq_url, "secret-key-123")
    assert words in str(err.value) and "secret-key-123" not in str(err.value)


def test_unreachable_server_is_reported():
    with pytest.raises(ValueError, match="Could not reach"):
        groq_chat([], "m", "http://127.0.0.1:9/openai/v1", "k", timeout=2)


def test_general_mode_answers_with_sources_and_caches_the_retriever():
    seen = []

    def fake_chat(messages):
        seen.append(messages)
        return {"answer": "Delaware law applies [1]."}

    mode = GeneralMode(chat_fn=fake_chat, embedder=HashEmbedder())
    rec = {"doc": DOC}
    res = mode.ask(rec, "Which law governs?")
    assert res["found"] and res["cited"] == [1] and res["passages"][0]["pages"] == [2]
    assert "Delaware" in seen[0][1]["content"]
    first = rec["retriever"]
    mode.ask(rec, "Which court?")
    assert rec["retriever"] is first


def test_availability_follows_the_environment(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert GeneralMode().available() is False
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert GeneralMode().available() is True
    assert "x" not in json.dumps(GeneralMode().info())


def test_real_embedding_model_if_available():
    from documind.general.qa import SentenceEmbedder
    try:
        emb = SentenceEmbedder("BAAI/bge-small-en-v1.5")
    except Exception as exc:  # no network and no cached model
        pytest.skip(f"embedding model not available: {type(exc).__name__}")
    hits = Retriever(DOC, embedder=emb).search("Which law governs this agreement?", k=3)
    assert any("Delaware" in h["text"] for h in hits[:2])