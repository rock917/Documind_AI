from fastapi.testclient import TestClient

from documind.api.app import create_app


class Finder:
    threshold = 8.69
    questions = {"Governing Law": "q1", "Cap On Liability": "q2"}

    def analyze(self, doc, categories=None, progress=None):
        return []


def test_about_lists_the_clause_types_and_threshold(tmp_path):
    client = TestClient(create_app(Finder(), str(tmp_path / "storage")))
    body = client.get("/api/about").json()
    assert body == {"categories": ["Governing Law", "Cap On Liability"], "threshold": 8.69}
