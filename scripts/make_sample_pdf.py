"""Turn one CUAD test-split contract (plain text) into a PDF, so it can be used as the web app's sample.

List candidates:   python -m scripts.make_sample_pdf --list
Build a PDF:       python -m scripts.make_sample_pdf --index 5
                   (or --title "exact title"). Writes samples_src\\<name>.pdf
CUAD is released under CC BY 4.0 (The Atticus Project).
"""
import argparse
import json
import re
from pathlib import Path

import pymupdf

def load(data_dir):
    d = Path(data_dir)
    contracts = json.loads((d / "contracts.json").read_text(encoding="utf-8"))
    present = {}
    with open(d / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            present.setdefault(r["title"], set())
            if r.get("answers"):
                present[r["title"]].add(r["category"])
    return contracts, present


def candidates(contracts, present, lo, hi):
    rows = []
    for title, cats in present.items():
        text = contracts.get(title)
        if not text:
            continue
        words = len(text.split())
        if lo <= words <= hi:
            rows.append((title, words, len(cats)))
    rows.sort(key=lambda r: (-r[2], r[1]))
    return rows


_ASCII = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2022": "*", "\u00a0": " ", "\u2026": "..."}


def _clean(s):
    for k, v in _ASCII.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def _wrap(font, words, size, width):
    lines, cur = [], ""
    for w in words:
        while font.text_length(w, fontsize=size) > width:  # a single over-long word: split it
            k = max(1, int(len(w) * width / font.text_length(w, fontsize=size)))
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(w[:k])
            w = w[k:]
        trial = (cur + " " + w) if cur else w
        if font.text_length(trial, fontsize=size) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def to_pdf(title, text):
    """Plain A4 pages, laid out line by line, so no words can be lost."""
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n|\r\n\s*\r\n", _clean(text))]
    paras = [p for p in paras if p]
    if len(paras) < 5:  # no blank lines: fall back to one paragraph per line
        paras = [" ".join(p.split()) for p in _clean(text).splitlines() if p.strip()]
    font = pymupdf.Font("tiro")
    size, lead, gap, margin = 10, 13.5, 6, 56
    rect = pymupdf.paper_rect("a4")
    width, bottom = rect.width - 2 * margin, rect.height - 60
    doc = pymupdf.open()
    page, y = None, 0

    def put(line, fontname="tiro", fs=size, extra=0):
        nonlocal page, y
        if page is None or y + lead > bottom:
            page = doc.new_page(width=rect.width, height=rect.height)
            y = margin
        page.insert_text((margin, y + fs), line, fontname=fontname, fontsize=fs)
        y += lead + extra

    for line in _wrap(font, _clean(title).split(), size, width):
        put(line, "tibo", 12)
    y += gap
    for para in paras:
        for line in _wrap(font, para.split(), size, width):
            put(line)
        y += gap
    return doc.tobytes(garbage=3, deflate=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--list", action="store_true")
    p.add_argument("--index", type=int)
    p.add_argument("--title")
    p.add_argument("--min-words", type=int, default=1500)
    p.add_argument("--max-words", type=int, default=6000)
    p.add_argument("--name", default="cuad-sample")
    args = p.parse_args()

    contracts, present = load(args.data_dir)
    rows = candidates(contracts, present, args.min_words, args.max_words)
    if args.list or (args.index is None and not args.title):
        print(f"{len(rows)} test contracts between {args.min_words} and {args.max_words} words (most clause types first):")
        for i, (t, w, c) in enumerate(rows[:25]):
            print(f"{i:2d}  {w:5d} words  {c:2d} clause types  {t[:70]}")
        print("Next: python -m scripts.make_sample_pdf --index <number>")
        return
    if args.title:
        title = args.title
        if title not in contracts:
            raise SystemExit("STOP: that title is not in contracts.json.")
    else:
        if not 0 <= args.index < len(rows):
            raise SystemExit("STOP: index out of range.")
        title = rows[args.index][0]
    data = to_pdf(title, contracts[title])
    out = Path("samples_src")
    out.mkdir(exist_ok=True)
    path = out / f"{args.name}.pdf"
    path.write_bytes(data)
    doc = pymupdf.open(stream=data, filetype="pdf")
    print(f"built {path}: {doc.page_count} pages from '{title}'")


if __name__ == "__main__":
    main()