import json

import pytest
import torch
from transformers import BertConfig, BertForQuestionAnswering, BertTokenizerFast

from documind.pdf.extract import build_document
from documind.pipeline.analyze import ClauseFinder, load_settings

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon "
         "agreement shall governed law delaware term year").split()
QUESTIONS = {"Governing Law": "Which law governs the agreement?", "Term": "How long is the term?", "Parties": "Who are the parties?"}


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_qa")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(set(WORDS)) + ["which", "who", "how", "long", "is", "the", "are"]
    (d / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    BertTokenizerFast(str(d / "vocab.txt")).save_pretrained(d)
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=512)
    BertForQuestionAnswering(cfg).save_pretrained(d)
    return str(d)


def make_doc(n_words=150):
    words = [WORDS[(i * 7) % len(WORDS)] for i in range(n_words)]
    half = n_words // 2
    pages = [[(10 + j, 10 + (j // 10) * 12, 20 + j, 20 + (j // 10) * 12, w, 0, j // 10, j % 10) for j, w in enumerate(words[:half])],
             [(10 + j, 10 + (j // 10) * 12, 20 + j, 20 + (j // 10) * 12, w, 0, j // 10, j % 10) for j, w in enumerate(words[half:])]]
    return build_document(pages, [[595.0, 842.0], [595.0, 842.0]])


def finder(tiny_dir, threshold=0.0, top_k=3):
    return ClauseFinder(tiny_dir, tiny_dir, QUESTIONS, threshold, device="cpu", max_length=64, stride=16, batch_size=4, top_k=top_k)


def test_passages_are_exact_slices_with_pages(tiny_dir):
    doc = make_doc()
    results = finder(tiny_dir).analyze(doc)
    assert [r["category"] for r in results] == list(QUESTIONS)
    for r in results:
        assert r["n_windows"] > 1
        assert 1 <= len(r["passages"]) <= 3
        for p in r["passages"]:
            assert doc["text"][p["char_start"]:p["char_end"]] == p["text"]
            assert p["pages"] and set(p["pages"]) <= {1, 2}
            assert all(len(rect) == 4 for h in p["highlights"] for rect in h["rects"])
        scores = [p["score"] for p in r["passages"]]
        assert scores == sorted(scores, reverse=True)
        assert r["present_score"] == scores[0]


def test_threshold_decides_found_or_not_found(tiny_dir):
    doc = make_doc()
    assert all(r["present"] for r in finder(tiny_dir, threshold=-1e9).analyze(doc))
    assert not any(r["present"] for r in finder(tiny_dir, threshold=1e9).analyze(doc))
    r = finder(tiny_dir, threshold=2.5).analyze(doc, ["Term"])[0]
    assert r["margin"] == pytest.approx(r["present_score"] - 2.5)


def test_top_k_progress_and_category_subset(tiny_dir):
    calls = []
    results = finder(tiny_dir, top_k=1).analyze(make_doc(), ["Parties", "Term"], progress=lambda d, t, c: calls.append((d, t, c)))
    assert [r["category"] for r in results] == ["Parties", "Term"]
    assert all(len(r["passages"]) == 1 for r in results)
    assert calls == [(1, 2, "Parties"), (2, 2, "Term")]


def test_errors_for_scan_and_unknown_category(tiny_dir):
    f = finder(tiny_dir)
    with pytest.raises(ValueError, match="OCR"):
        f.analyze(build_document([[], []]))
    with pytest.raises(ValueError, match="Unknown category"):
        f.analyze(make_doc(), ["No Such Category"])


def test_from_settings(tiny_dir, tmp_path):
    q = tmp_path / "q.json"
    q.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    s = tmp_path / "inf.json"
    s.write_text(json.dumps({"model_dir": tiny_dir, "tokenizer": tiny_dir, "questions_path": str(q), "threshold": 1.5,
                             "top_k": 2, "max_length": 64, "stride": 16, "batch_size": 4}), encoding="utf-8")
    f = ClauseFinder.from_settings(str(s), device="cpu")
    assert f.threshold == 1.5 and f.top_k == 2 and set(f.questions) == set(QUESTIONS)
    assert load_settings(str(s))["threshold"] == 1.5