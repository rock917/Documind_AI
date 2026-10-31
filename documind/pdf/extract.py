"""PDF text extraction with an exact map from character offsets to pages and page coordinates.

The model reads ONE plain string: all words of the PDF joined by single spaces. For every word we remember its
character range in that string, its page and its box on the page. A predicted span (char_start, char_end) can then be
turned into page numbers and highlight rectangles without any fuzzy text matching.
"""
from bisect import bisect_left, bisect_right

import pymupdf

SCANNED_WORDS_PER_PAGE = 20  # fewer words per page on average -> probably a scan that needs OCR


def build_document(pages_words, page_sizes=None):
    """Build the text and the word table from words grouped by page.

    pages_words: one list per page of PyMuPDF word tuples (x0, y0, x1, y1, word, block_no, line_no, word_no).
    """
    parts, words = [], []
    pos = 0
    page_ranges = []
    for page_no, page_words in enumerate(pages_words, start=1):
        first_word = len(words)
        for w in page_words:
            token = str(w[4]).strip()
            if not token:
                continue
            if parts:
                pos += 1  # the joining space
            start = pos
            pos += len(token)
            parts.append(token)
            words.append({"start": start, "end": pos, "page": page_no, "bbox": [float(w[0]), float(w[1]), float(w[2]), float(w[3])],
                          "line": [page_no, int(w[5]), int(w[6])]})
        used = words[first_word:]
        page_ranges.append({"page": page_no, "n_words": len(used),
                            "char_start": used[0]["start"] if used else None,
                            "char_end": used[-1]["end"] if used else None})
    n_pages = len(pages_words)
    return {
        "text": " ".join(parts),
        "words": words,
        "word_starts": [w["start"] for w in words],
        "pages": page_ranges,
        "page_sizes": page_sizes or [],
        "n_pages": n_pages,
        "n_words": len(words),
        "likely_scanned": n_pages > 0 and len(words) / n_pages < SCANNED_WORDS_PER_PAGE,
    }


def span_to_locations(doc, char_start, char_end):
    """Pages and highlight boxes covered by the character span [char_start, char_end) of doc['text'].

    Returns {"pages": [...], "highlights": [{"page": p, "rects": [[x0, y0, x1, y1], ...]}]} with one rectangle per
    text line. An empty or out-of-range span returns no pages.
    """
    words, starts = doc["words"], doc["word_starts"]
    if not words or char_end <= char_start:
        return {"pages": [], "highlights": []}
    first = max(bisect_right(starts, char_start) - 1, 0)
    while first < len(words) and words[first]["end"] <= char_start:
        first += 1
    last = bisect_left(starts, char_end) - 1
    if first > last or first >= len(words):
        return {"pages": [], "highlights": []}
    lines = {}  # (page, block, line) -> merged box, in reading order
    for w in words[first:last + 1]:
        key = tuple(w["line"])
        x0, y0, x1, y1 = w["bbox"]
        if key in lines:
            b = lines[key]
            lines[key] = [min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1)]
        else:
            lines[key] = [x0, y0, x1, y1]
    by_page = {}
    for (page, _block, _line), rect in lines.items():
        by_page.setdefault(page, []).append(rect)
    pages = sorted(by_page)
    return {"pages": pages, "highlights": [{"page": p, "rects": by_page[p]} for p in pages]}


def extract_pdf(path):
    """Read a PDF file into the document structure of build_document. Raises ValueError for unreadable or locked files."""
    try:
        pdf = pymupdf.open(path)
    except Exception as exc:
        raise ValueError("This file is not a readable PDF.") from exc
    with pdf:
        if pdf.needs_pass:
            raise ValueError("This PDF is password protected.")
        pages_words, sizes = [], []
        for page in pdf:
            pages_words.append(page.get_text("words", sort=True))
            sizes.append([float(page.rect.width), float(page.rect.height)])
    return build_document(pages_words, sizes)


def main():
    """Check a real PDF:  python -m documind.pdf.extract path\\to\\file.pdf [--find "some phrase"]"""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("pdf")
    p.add_argument("--find", default=None, help="a phrase to locate; prints its pages and highlight boxes")
    args = p.parse_args()
    doc = extract_pdf(args.pdf)
    print(f"pages: {doc['n_pages']} | words: {doc['n_words']} | characters: {len(doc['text'])} | likely scanned: {doc['likely_scanned']}")
    for pg in doc["pages"][:5]:
        print(f"  page {pg['page']}: {pg['n_words']} words")
    if doc["n_pages"] > 5:
        print(f"  ... ({doc['n_pages'] - 5} more pages)")
    print("text starts:", doc["text"][:300].replace("\n", " "))
    if args.find:
        i = doc["text"].lower().find(args.find.lower())
        if i < 0:
            print(f"phrase not found: {args.find!r}")
        else:
            loc = span_to_locations(doc, i, i + len(args.find))
            print(f"phrase found at characters {i}-{i + len(args.find)} on page(s) {loc['pages']}")
            for h in loc["highlights"]:
                print(f"  page {h['page']}: {len(h['rects'])} line box(es), first box {[round(v, 1) for v in h['rects'][0]]}")


if __name__ == "__main__":
    main()