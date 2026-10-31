"""Find clauses in one document: run the QA model once per category and attach pages and highlight boxes.

For each category the model reads the whole document in overlapping windows and returns ranked candidate passages
with a present_score (best passage score minus the 'no answer' score). A category counts as FOUND when the best
present_score reaches the threshold that was chosen on the validation split; otherwise the answer is 'not found'.

Check a real PDF from the project root (E:\\Documind):
    python -m documind.pipeline.analyze path\\to\\contract.pdf --categories "Governing Law,Parties"
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from documind.data.windows import make_windows
from documind.eval.predict import finite_or_none, window_logits
from documind.eval.qa_decode import aggregate_windows, decode_window
from documind.pdf.extract import span_to_locations


def load_settings(path="configs/inference.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class ClauseFinder:
    def __init__(self, model_dir, tokenizer, questions, threshold, device=None,
                 max_length=384, stride=128, batch_size=16, top_k=3):
        self.tok = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        self.pad_id = self.model.config.pad_token_id or 0
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            self.model.half()
        self.model.to(self.device)
        self.questions = questions
        self.threshold = float(threshold)
        self.max_length, self.stride, self.batch_size, self.top_k = max_length, stride, batch_size, top_k

    @classmethod
    def from_settings(cls, settings_path="configs/inference.json", device=None):
        s = load_settings(settings_path)
        questions = json.loads(Path(s["questions_path"]).read_text(encoding="utf-8"))
        return cls(s["model_dir"], s["tokenizer"], questions, s["threshold"], device,
                   s["max_length"], s["stride"], s["batch_size"], s["top_k"])

    def find(self, doc, category):
        """Result for one category: found or not, the score, and the top passages with pages and highlight boxes."""
        if category not in self.questions:
            raise ValueError(f"Unknown category: {category!r}")
        text = doc["text"]
        windows = make_windows(self.tok, self.questions[category], text, [], self.max_length, self.stride)
        logits = window_logits(self.model, windows, self.batch_size, self.device, self.pad_id)
        decoded = [decode_window(s, e, w["offset_mapping"], top_spans=self.top_k) for (s, e), w in zip(logits, windows)]
        agg = aggregate_windows(decoded, text, k=self.top_k)
        passages = []
        for c in agg["candidates"]:
            loc = span_to_locations(doc, c["char_start"], c["char_end"])
            passages.append({"text": c["text"], "char_start": c["char_start"], "char_end": c["char_end"],
                             "score": finite_or_none(c["present_score"]),
                             "pages": loc["pages"], "highlights": loc["highlights"]})
        score = finite_or_none(agg["present_score"])
        return {"category": category,
                "present": score is not None and score >= self.threshold,
                "present_score": score,
                "threshold": self.threshold,
                "margin": None if score is None else score - self.threshold,
                "n_windows": len(windows),
                "passages": passages}

    def analyze(self, doc, categories=None, progress=None):
        """Run every category (or the given ones) on a document from documind.pdf.extract. progress(done, total, category)."""
        if doc["n_words"] == 0:
            raise ValueError("This document has no extractable text (it is probably a scan and needs OCR).")
        cats = list(categories) if categories else list(self.questions)
        results = []
        for i, cat in enumerate(cats, start=1):
            results.append(self.find(doc, cat))
            if progress:
                progress(i, len(cats), cat)
        return results


def main():
    import argparse
    import time

    from documind.pdf.extract import extract_pdf

    p = argparse.ArgumentParser()
    p.add_argument("pdf")
    p.add_argument("--categories", default=None, help="comma-separated category names (default: all 41)")
    p.add_argument("--settings", default="configs/inference.json")
    args = p.parse_args()

    doc = extract_pdf(args.pdf)
    print(f"document: {doc['n_pages']} pages, {doc['n_words']} words")
    finder = ClauseFinder.from_settings(args.settings)
    print(f"device: {finder.device} | threshold: {finder.threshold}")
    cats = [c.strip() for c in args.categories.split(",")] if args.categories else None
    start = time.time()
    results = finder.analyze(doc, cats)
    print(f"analysed {len(results)} categories in {time.time() - start:.1f} s\n")
    for r in sorted(results, key=lambda r: -(r["present_score"] if r["present_score"] is not None else -1e9)):
        tag = "FOUND    " if r["present"] else "not found"
        score = "n/a" if r["present_score"] is None else f"{r['present_score']:.2f}"
        line = f"{tag} {r['category']:<38} score {score:>7}"
        if r["passages"]:
            top = r["passages"][0]
            snippet = top["text"].replace("\n", " ")[:90]
            line += f" | p.{top['pages']} | {snippet}"
        print(line)


if __name__ == "__main__":
    main()