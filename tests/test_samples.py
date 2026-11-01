import json

import pymupdf
from fastapi.testclient import TestClient

from documind.api.app import create_app


class Finder:
    threshold = 8.69
    questions = {"Governing Law": "q1", "Parties": "q2"}

    def analyze(self, doc, categories=None, progress=None):
        raise AssertionError("a sample must never be analysed again")


def make_sample(root, name="demo"):
    folder = root / name
    folder.mkdir(parents=True)
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "This Agreement shall be governed by the laws of Delaware.", fontsize=11)
    pdf.save(str(folder / "original.pdf"))
    pdf.close()
    results = [{"category": "Governing Law", "present": True, "present_score": 15.0, "threshold": 8.69, "margin": 6.3, "n_windows": 1,
                "passages": [{"text": "governed by the laws of Delaware", "char_start": 23, "char_end": 55, "score": 15.0, "pages": [1], "highlights": []}]},
               {"category": "Parties", "present": False, "present_score": 1.0, "threshold": 8.69, "margin": -7.7, "n_windows": 1, "passages": []}]
    (folder / "results.json").write_text(json.dumps({"filename": "demo.pdf", "seconds": 4.2, "threshold": 8.69, "results": results}), encoding="utf-8")


def test_sample_is_listed_and_opens_without_analysis(tmp_path):
    make_sample(tmp_path / "samples")
    client = TestClient(create_app(Finder(), str(tmp_path / "storage"), samples_dir=str(tmp_path / "samples")))
    listed = client.get("/api/samples").json()["samples"]
    assert listed == [{"id": "sample-demo", "filename": "demo.pdf", "n_pages": 1}]
    info = client.get("/api/documents/sample-demo").json()
    assert info["state"] == "done" and info["sample"] is True and info["seconds"] == 4.2
    body = client.get("/api/documents/sample-demo/results").json()
    assert [r["category"] for r in body["results"]] == ["Governing Law", "Parties"]
    assert body["results"][0]["present"] is True
    assert client.get("/api/documents/sample-demo/page/1.png").status_code == 200
    assert client.get("/api/documents/sample-demo/report.pdf").status_code == 200


def test_no_samples_folder_gives_an_empty_list(tmp_path):
    client = TestClient(create_app(Finder(), str(tmp_path / "storage"), samples_dir=str(tmp_path / "nothing")))
    assert client.get("/api/samples").json() == {"samples": []}


def test_broken_sample_is_skipped(tmp_path):
    make_sample(tmp_path / "samples", "good")
    bad = tmp_path / "samples" / "bad"
    bad.mkdir()
    (bad / "original.pdf").write_bytes(b"not a pdf")
    (bad / "results.json").write_text("{", encoding="utf-8")
    client = TestClient(create_app(Finder(), str(tmp_path / "storage"), samples_dir=str(tmp_path / "samples")))
    assert [s["id"] for s in client.get("/api/samples").json()["samples"]] == ["sample-good"]
