"""Compare OCR output with the real text layer: how many of the original words does OCR recover, and how fast."""
import re
import sys
import time
from collections import Counter
from pathlib import Path

from documind.pdf.extract import extract_pdf
from documind.pdf.prepare import ocr_pdf

FOLDER = Path("data/ocr_check")


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def main():
    text_pdf, scan_pdf, out_pdf = FOLDER / "sample_text.pdf", FOLDER / "sample_scan.pdf", FOLDER / "sample_scan_ocr.pdf"
    for p in (text_pdf, scan_pdf):
        if not p.exists():
            sys.exit(f"Missing {p}. Run scripts\\make_scan_pair.py first.")
    start = time.time()
    ocr_pdf(scan_pdf, out_pdf)
    seconds = time.time() - start
    truth = extract_pdf(text_pdf)
    got = extract_pdf(out_pdf)
    t, g = Counter(tokens(truth["text"])), Counter(tokens(got["text"]))
    hit = sum(min(n, g[w]) for w, n in t.items())
    total = sum(t.values())
    extra = sum(g.values()) - hit
    print(f"pages: {truth['n_pages']} | OCR time: {seconds:.1f} s ({seconds / truth['n_pages']:.1f} s per page)")
    print(f"original words: {total} | OCR words: {sum(g.values())}")
    print(f"recovered (word recall): {hit / total:.3f} | extra or garbled words in OCR: {extra} ({extra / max(1, sum(g.values())):.3f} of OCR output)")


main()