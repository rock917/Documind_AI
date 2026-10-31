import pymupdf
import pytest
from fastapi.testclient import TestClient

from documind.api.app import create_app
from documind.general.qa import GeneralMode

SENTENCE = "This Agreement shall be governed by the laws of Delaware and nothing else applies."


class CountingFinder:
    threshold = 8.69
    questions = {"Governing Law": "q"}
    calls = 0

    def analyze(self, doc, categories=None, progress=None):
        CountingFinder.calls += 1
        return []


def pdf_bytes(tmp_path):
    pdf = pymupdf.open()
    for text in ["Intro page about apples and oranges and pears. " * 40,
                 "Filler words about rivers. " * 40 + SENTENCE + " " + "Filler words about rivers. " * 15]:
        pdf.new_page().insert_textbox(pymupdf.Rect(60, 60, 540, 780), text, fontsize=10)
    path = tmp_path / "doc.pdf"
    pdf.save(str(path))
    return path.read_bytes()


def fake_chat(messages):
    return {"answer": "Delaware law governs [1]."}


def make_client(tmp_path, general):
    return TestClient(create_app(CountingFinder(), tmp_path / "storage", web_dir=tmp_path / "noweb", general=general))


def upload(client, tmp_path, analyze=False):
    r = client.post("/api/documents", params={"analyze": str(analyze).lower()}, files={"file": ("d.pdf", pdf_bytes(tmp_path), "application/pdf")})
    assert r.status_code == 200
    return r.json()


def test_general_upload_skips_the_clause_analysis(tmp_path):
    CountingFinder.calls = 0
    client = make_client(tmp_path, GeneralMode(chat_fn=fake_chat, use_dense=False))
    info = upload(client, tmp_path, analyze=False)
    assert info["analyze"] is False
    import time
    for _ in range(100):
        if client.get(f"/api/documents/{info['id']}").json()["state"] == "done":
            break
        time.sleep(0.05)
    res = client.get(f"/api/documents/{info['id']}/results").json()
    assert res["results"] == [] and CountingFinder.calls == 0


def test_general_info_reflects_availability(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert make_client(tmp_path, GeneralMode()).get("/api/general").json()["available"] is False
    assert make_client(tmp_path, None).get("/api/general").json()["available"] is False
    on = make_client(tmp_path, GeneralMode(chat_fn=fake_chat, use_dense=False)).get("/api/general").json()
    assert on["available"] is True and on["key_variable"] == "GROQ_API_KEY"


def test_ask_needs_a_key_permission_and_a_sensible_question(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    off = make_client(tmp_path, GeneralMode())
    doc_id = upload(off, tmp_path)["id"]
    assert off.post(f"/api/documents/{doc_id}/ask", json={"question": "q", "allow_network": True}).status_code == 503
    client = make_client(tmp_path, GeneralMode(chat_fn=fake_chat, use_dense=False))
    doc_id = upload(client, tmp_path)["id"]
    url = f"/api/documents/{doc_id}/ask"
    assert client.post(url, json={"question": "Which law?"}).status_code == 400  # no permission given
    assert client.post(url, json={"question": "   ", "allow_network": True}).status_code == 400
    assert client.post(url, json={"question": "x" * 501, "allow_network": True}).status_code == 400
    assert client.post("/api/documents/nope/ask", json={"question": "q", "allow_network": True}).status_code == 404


def test_ask_returns_answer_cited_sources_and_pages(tmp_path):
    client = make_client(tmp_path, GeneralMode(chat_fn=fake_chat, use_dense=False))
    doc_id = upload(client, tmp_path)["id"]
    r = client.post(f"/api/documents/{doc_id}/ask", json={"question": "Which law governs the agreement?", "allow_network": True})
    assert r.status_code == 200
    out = r.json()
    assert out["found"] is True and out["cited"] == [1] and "Delaware" in out["answer"]
    top = out["passages"][0]
    assert "Delaware" in top["text"] and top["pages"] == [2] and top["highlights"]
    page = client.get(f"/api/documents/{doc_id}/page/2.png", params={"hl": f"{top['char_start']}-{top['char_end']}"})
    assert page.status_code == 200


def test_groq_failures_are_reported_as_a_bad_gateway(tmp_path):
    def broken(messages):
        raise ValueError("Groq rejected the API key. Check the GROQ_API_KEY environment variable.")

    client = make_client(tmp_path, GeneralMode(chat_fn=broken, use_dense=False))
    doc_id = upload(client, tmp_path)["id"]
    r = client.post(f"/api/documents/{doc_id}/ask", json={"question": "Which law?", "allow_network": True})
    assert r.status_code == 502 and "rejected" in r.json()["detail"]
