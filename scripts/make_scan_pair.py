"""Make a text copy and an image-only "scanned" copy of the first pages of a PDF, to test OCR side by side."""
import sys
from pathlib import Path

import pymupdf

PAGES = 6
DPI = 150
OUT = Path("data/ocr_check")


def pick():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    home = Path.home()
    pdfs = sorted(p for d in (home / "Downloads", home / "Documents") if d.exists() for p in d.glob("*.pdf"))
    if not pdfs:
        sys.exit("No PDFs found in Downloads or Documents. Pass a path: python scripts\\make_scan_pair.py file.pdf")
    for i, p in enumerate(pdfs, 1):
        print(i, p.name)
    choice = input("Number of a printed, text-based contract PDF: ").strip()
    if not choice.isdigit() or not 1 <= int(choice) <= len(pdfs):
        sys.exit("Not a valid number.")
    return pdfs[int(choice) - 1]


def main():
    src_path = pick()
    OUT.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(str(src_path)) as src:
        n = min(PAGES, len(src))
        text_doc = pymupdf.open()
        text_doc.insert_pdf(src, from_page=0, to_page=n - 1)
        text_doc.save(str(OUT / "sample_text.pdf"))
        scan_doc = pymupdf.open()
        for i in range(n):
            page = src[i]
            pix = page.get_pixmap(dpi=DPI)
            sp = scan_doc.new_page(width=page.rect.width, height=page.rect.height)
            sp.insert_image(sp.rect, pixmap=pix)
        scan_doc.save(str(OUT / "sample_scan.pdf"))
    words = sum(len(p.get_text("words")) for p in pymupdf.open(str(OUT / "sample_text.pdf")))
    scan_words = sum(len(p.get_text("words")) for p in pymupdf.open(str(OUT / "sample_scan.pdf")))
    print(f"source: {src_path.name} | pages used: {n}")
    print(f"sample_text.pdf: {words} words | sample_scan.pdf: {scan_words} words (should be 0)")
    print(f"saved in {OUT.resolve()}")


main()