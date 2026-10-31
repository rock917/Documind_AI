import time

import pymupdf
import pytest
from fastapi.testclient import TestClient

from documind.api.app import create_app
from documind.pdf.prepare import find_soffice, find_tesseract

needs_soffice = pytest.mark.skipif(find_soffice() is None, reason="LibreOffice not installed")
needs_tesseract = pytest.mark.skipif(find_tesseract() is None, reason="Tesseract not installed")


class RecordingFinder:
    threshold = 8.69
    questions = {"Governing Law": "q"}
    seen = ""

    def analyze(self, doc, categories=None, progress=None):
        RecordingFinder.seen = doc["text"]
        if progress:
            progress(1, 1, "Governing Law")
        return [{"category": "Governing Law", "present": True, "present_score": 12.0, "threshold": 8.69, "margin": 3.3,
                 "n_windows": 1, "passages": []}]


def wait_done(client, doc_id, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        info = client.get(f"/api/documents/{doc_id}").json()
        if info["state"] in ("done", "error"):
            return info
        time.sleep(0.1)
    raise AssertionError("job did not finish")


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(RecordingFinder(), tmp_path / "storage", web_dir=tmp_path / "noweb"))


def scan_bytes(tmp_path):
    src = pymupdf.open()
    page = src.new_page(width=400, height=200)
    page.insert_text((30, 80), "This Agreement is governed by Delaware law.", fontsize=14)
    pix = page.get_pixmap(dpi=200)
    scan = pymupdf.open()
    sp = scan.new_page(width=400, height=200)
    sp.insert_image(sp.rect, pixmap=pix)
    path = tmp_path / "scan.pdf"
    scan.save(str(path))
    return path.read_bytes()


def test_page_sizes_are_exposed(client, tmp_path):
    pdf = pymupdf.open()
    pdf.new_page(width=300, height=500).insert_text((50, 50), "hello world " * 5)
    path = tmp_path / "t.pdf"
    pdf.save(str(path))
    info = client.post("/api/documents", files={"file": ("t.pdf", path.read_bytes(), "application/pdf")}).json()
    assert info["ocr"] is False
    assert [round(x) for x in info["page_sizes"][0]] == [300, 500]


@needs_soffice
def test_word_style_upload_is_converted_and_analysed(client):
    rtf = rb"{\rtf1\ansi This Agreement is governed by the laws of Delaware.\par}"
    r = client.post("/api/documents", files={"file": ("contract.rtf", rtf, "application/rtf")})
    assert r.status_code == 200 and r.json()["filename"] == "contract.rtf"
    info = wait_done(client, r.json()["id"])
    assert info["state"] == "done" and info["ocr"] is False
    assert "Delaware" in RecordingFinder.seen
    assert client.get(f"/api/documents/{info['id']}/page/1.png").status_code == 200


@needs_tesseract
def test_scanned_upload_is_read_with_ocr(client, tmp_path):
    r = client.post("/api/documents", files={"file": ("scan.pdf", scan_bytes(tmp_path), "application/pdf")})
    assert r.status_code == 200
    info = wait_done(client, r.json()["id"])
    assert info["state"] == "done" and info["ocr"] is True and info["n_words"] > 0
    assert "Delaware" in RecordingFinder.seen
    assert client.get(f"/api/documents/{info['id']}/page/1.png").status_code == 200


def test_empty_word_file_is_rejected(client):
    assert client.post("/api/documents", files={"file": ("e.docx", b"", "application/octet-stream")}).status_code == 400
