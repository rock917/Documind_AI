import time

import pymupdf
from fastapi.testclient import TestClient

from documind.api.app import create_app
from documind.report import build_report

INFO = {"filename": "deal <v2> & co.pdf", "n_pages": 3, "ocr": False}
RESULTS = [
    {"category": "Governing Law", "present": True, "present_score": 14.1, "margin": 5.4,
     "passages": [{"text": "This Agreement is governed by the laws of Delaware.", "pages": [6]}]},
    {"category": "Cap On Liability", "present": True, "present_score": 9.0, "margin": 0.3,
     "passages": [{"text": "Liability is capped at fees paid.", "pages": [9, 10]}]},
    {"category": "Non-Compete", "present": False, "present_score": -4.2, "margin": -12.9, "passages": []},
]


def text_of(pdf_bytes):
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    return doc, "\n".join(p.get_text() for p in doc)


def test_report_is_a_readable_pdf_with_found_and_missing_clauses():
    pdf = build_report(INFO, 8.69, RESULTS)
    assert pdf.startswith(b"%PDF-")
    doc, text = text_of(pdf)
    assert "deal <v2> & co.pdf" in text and "2 of 3" in text
    assert "Governing Law" in text and "Delaware" in text and "Page 6" in text
    assert text.count("Close call") == 1 and "Non-Compete (-4.2)" in text
    assert "page 1 of %d" % doc.page_count in text


def test_report_handles_nothing_found_and_many_clauses():
    _, text = text_of(build_report(INFO, 8.69, [dict(RESULTS[2])]))
    assert "No clause type reached the threshold." in text
    many = [dict(RESULTS[0], category="Clause %d" % i) for i in range(120)]
    doc, _ = text_of(build_report(INFO, 8.69, many))
    assert doc.page_count > 1


class Finder:
    threshold = 8.69
    questions = {"Governing Law": "q"}

    def analyze(self, doc, categories=None, progress=None):
        return RESULTS


def make_pdf():
    pdf = pymupdf.open()
    pdf.new_page().insert_textbox(pymupdf.Rect(60, 60, 540, 780), "This Agreement is governed by the laws of Delaware. " * 20, fontsize=10)
    return pdf.tobytes()


def wait_done(client, doc_id):
    for _ in range(100):
        if client.get("/api/documents/" + doc_id).json()["state"] in ("done", "error"):
            return
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_report_endpoint(tmp_path):
    client = TestClient(create_app(Finder(), str(tmp_path / "storage")))
    doc_id = client.post("/api/documents", files={"file": ("my deal.pdf", make_pdf(), "application/pdf")}).json()["id"]
    wait_done(client, doc_id)
    resp = client.get("/api/documents/%s/report.pdf" % doc_id)
    assert resp.status_code == 200 and resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF-") and "my_deal_report.pdf" in resp.headers["content-disposition"]
    assert client.get("/api/documents/nope/report.pdf").status_code == 404
    ask_id = client.post("/api/documents?analyze=false", files={"file": ("n.pdf", make_pdf(), "application/pdf")}).json()["id"]
    wait_done(client, ask_id)
    assert client.get("/api/documents/%s/report.pdf" % ask_id).status_code == 409
