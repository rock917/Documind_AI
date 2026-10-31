import pymupdf
import pytest

from documind.pdf.extract import extract_pdf
from documind.pdf.prepare import find_soffice, find_tesseract, ocr_pdf, office_to_pdf

needs_soffice = pytest.mark.skipif(find_soffice() is None, reason="LibreOffice not installed")
needs_tesseract = pytest.mark.skipif(find_tesseract() is None, reason="Tesseract not installed")


def test_missing_tools_give_plain_messages(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCUMIND_SOFFICE", str(tmp_path / "nope.exe"))
    monkeypatch.setenv("DOCUMIND_TESSERACT", str(tmp_path / "nope.exe"))
    with pytest.raises(ValueError, match="LibreOffice"):
        office_to_pdf(tmp_path / "a.docx", tmp_path)
    with pytest.raises(ValueError, match="Tesseract"):
        ocr_pdf(tmp_path / "a.pdf", tmp_path / "b.pdf")


@needs_soffice
def test_word_style_file_becomes_a_readable_pdf(tmp_path):
    src = tmp_path / "sample.rtf"
    src.write_text(r"{\rtf1\ansi This Agreement is governed by the laws of Delaware.\par}")
    pdf = office_to_pdf(src, tmp_path)
    doc = extract_pdf(pdf)
    assert "governed by the laws of Delaware" in doc["text"]


@needs_soffice
def test_missing_source_file_is_a_value_error(tmp_path):
    with pytest.raises(ValueError):
        office_to_pdf(tmp_path / "does_not_exist.docx", tmp_path)


def make_scan(path):
    """A one-page PDF that holds only a picture of text (no text layer), like a scanner would make."""
    src = pymupdf.open()
    page = src.new_page(width=400, height=200)
    page.insert_text((30, 80), "This Agreement is governed by Delaware law.", fontsize=14)
    pix = page.get_pixmap(dpi=200)
    scan = pymupdf.open()
    sp = scan.new_page(width=400, height=200)
    sp.insert_image(sp.rect, pixmap=pix)
    scan.save(str(path))


@needs_tesseract
def test_ocr_adds_a_text_layer_and_keeps_the_page_size(tmp_path):
    scan, out = tmp_path / "scan.pdf", tmp_path / "ocr.pdf"
    make_scan(scan)
    assert extract_pdf(scan)["n_words"] == 0
    seen = []
    ocr_pdf(scan, out, progress=lambda done, total: seen.append((done, total)))
    doc = extract_pdf(out)
    assert "Delaware" in doc["text"]
    assert seen == [(1, 1)]
    w, h = doc["page_sizes"][0]
    assert abs(w - 400) < 2 and abs(h - 200) < 2


@needs_tesseract
def test_ocr_refuses_too_many_pages(tmp_path):
    scan = tmp_path / "scan.pdf"
    make_scan(scan)
    with pytest.raises(ValueError, match="limited"):
        ocr_pdf(scan, tmp_path / "o.pdf", max_pages=0)