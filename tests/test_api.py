import time

import pymupdf
import pytest
from fastapi.testclient import TestClient

from documind.api.app import create_app, parse_spans


class FakeFinder:
    """Stands in for the model: no GPU, deterministic results."""
    threshold = 8.69
    questions = {"Governing Law": "q1", "Parties": "q2"}

    def analyze(self, doc, categories=None, progress=None):
        text = doc["text"]
        start = text.index("governed by")
        end = start + len("governed by the laws of Delaware")
        out = []
        for i, cat in enumerate(self.questions, start=1):
            out.append({"category": cat, "present": cat == "Governing Law", "present_score": 15.0 if cat == "Governing Law" else 1.0,
                        "threshold": self.threshold, "margin": 6.3, "n_windows": 2,
                        "passages": [{"text": text[start:end], "char_start": start, "char_end": end, "score": 15.0, "pages": [2], "highlights": []}]})
            if progress:
                progress(i, len(self.questions), cat)
        return out


class BoomFinder(FakeFinder):
    def analyze(self, doc, categories=None, progress=None):
        raise RuntimeError("model exploded")


def make_pdf_bytes(tmp_path, pages):
    pdf = pymupdf.open()
    for lines in pages:
        page = pdf.new_page()
        y = 72
        for line in lines:
            page.insert_text((72, y), line, fontsize=11)
            y += 16
    path = tmp_path / "t.pdf"
    pdf.save(str(path))
    pdf.close()
    return path.read_bytes()


GOOD = [["This Agreement is made on January 5, 2020", "between Acme Corp and Beta LLC."],
        ["This Agreement shall be governed by the laws of Delaware."]]


def wait_done(client, doc_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        info = client.get(f"/api/documents/{doc_id}").json()
        if info["state"] in ("done", "error"):
            return info
        time.sleep(0.05)
    raise AssertionError("job did not finish")


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(FakeFinder(), tmp_path / "storage", web_dir=tmp_path / "noweb"))


def upload(client, data, name="c.pdf"):
    return client.post("/api/documents", files={"file": (name, data, "application/pdf")})


def test_upload_progress_and_results(client, tmp_path):
    r = upload(client, make_pdf_bytes(tmp_path, GOOD))
    assert r.status_code == 200
    info = r.json()
    assert info["n_pages"] == 2 and info["total"] == 2 and info["filename"] == "c.pdf"
    done = wait_done(client, info["id"])
    assert done["state"] == "done" and done["done"] == 2 and done["seconds"] is not None
    res = client.get(f"/api/documents/{info['id']}/results").json()
    assert res["threshold"] == 8.69 and [x["category"] for x in res["results"]] == ["Governing Law", "Parties"]
    assert (tmp_path / "storage" / info["id"] / "results.json").exists()


def test_page_image_with_and_without_highlight(client, tmp_path):
    info = upload(client, make_pdf_bytes(tmp_path, GOOD)).json()
    wait_done(client, info["id"])
    res = client.get(f"/api/documents/{info['id']}/results").json()
    p = res["results"][0]["passages"][0]
    plain = client.get(f"/api/documents/{info['id']}/page/2.png")
    marked = client.get(f"/api/documents/{info['id']}/page/2.png", params={"hl": f"{p['char_start']}-{p['char_end']}"})
    assert plain.status_code == marked.status_code == 200
    assert plain.content[:8] == b"\x89PNG\r\n\x1a\n" and marked.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert plain.content != marked.content  # the highlight changed the picture
    other_page = client.get(f"/api/documents/{info['id']}/page/1.png", params={"hl": f"{p['char_start']}-{p['char_end']}"})
    assert other_page.content == client.get(f"/api/documents/{info['id']}/page/1.png").content  # span is on page 2 only


def test_bad_inputs(client, tmp_path, monkeypatch):
    monkeypatch.setenv("DOCUMIND_TESSERACT", "no-such-tesseract")  # this test covers the case where OCR is not available
    assert upload(client, b"hello", "notes.txt").status_code == 400
    assert upload(client, b"not really a pdf", "fake.pdf").status_code == 400
    assert upload(client, b"%PDF-1.4 garbage", "broken.pdf").status_code == 422
    assert upload(client, make_pdf_bytes(tmp_path, [[]]), "blank.pdf").status_code == 422  # no text: scan
    info = upload(client, make_pdf_bytes(tmp_path, GOOD)).json()
    assert client.get("/api/documents/nope").status_code == 404
    assert client.get(f"/api/documents/{info['id']}/page/9.png").status_code == 404
    assert client.get(f"/api/documents/{info['id']}/page/1.png", params={"hl": "abc"}).status_code == 400
    assert client.get(f"/api/documents/{info['id']}/page/1.png", params={"hl": "50-10"}).status_code == 400


def test_results_before_done_and_failed_job(tmp_path):
    client = TestClient(create_app(BoomFinder(), tmp_path / "s2", web_dir=tmp_path / "noweb"))
    info = upload(client, make_pdf_bytes(tmp_path, GOOD)).json()
    done = wait_done(client, info["id"])
    assert done["state"] == "error" and "model exploded" in done["error"]
    r = client.get(f"/api/documents/{info['id']}/results")
    assert r.status_code == 500 and "model exploded" in r.json()["detail"]


def test_home_without_web_page_and_parse_spans(client):
    assert "DocuMind API" in client.get("/").json()["message"]
    assert parse_spans(["3-9", "10-12"]) == [(3, 9), (10, 12)]