import pymupdf
import pytest

from documind.pdf.extract import build_document, extract_pdf, span_to_locations


def w(x0, y0, x1, y1, word, block=0, line=0, n=0):
    return (x0, y0, x1, y1, word, block, line, n)


PAGES = [
    [w(10, 10, 40, 20, "This", 0, 0, 0), w(45, 10, 90, 20, "Agreement", 0, 0, 1), w(10, 30, 50, 40, "is", 0, 1, 0)],
    [w(10, 10, 60, 20, "governed", 0, 0, 0), w(65, 10, 80, 20, "by", 0, 0, 1), w(85, 10, 120, 20, "Delaware", 0, 0, 2)],
]


def test_text_and_offsets_are_consistent():
    doc = build_document(PAGES)
    assert doc["text"] == "This Agreement is governed by Delaware"
    for word in doc["words"]:
        assert doc["text"][word["start"]:word["end"]] in {"This", "Agreement", "is", "governed", "by", "Delaware"}
    assert [doc["text"][x["start"]:x["end"]] for x in doc["words"]][3] == "governed"
    assert doc["pages"][0]["char_start"] == 0 and doc["pages"][1]["char_end"] == len(doc["text"])


def test_span_inside_one_page_gives_one_rect_per_line():
    doc = build_document(PAGES)
    start = doc["text"].index("Agreement")
    loc = span_to_locations(doc, start, start + len("Agreement is"))
    assert loc["pages"] == [1]
    assert loc["highlights"][0]["rects"] == [[45.0, 10.0, 90.0, 20.0], [10.0, 30.0, 50.0, 40.0]]


def test_span_across_pages_returns_both_pages():
    doc = build_document(PAGES)
    start = doc["text"].index("is governed")
    loc = span_to_locations(doc, start, start + len("is governed"))
    assert loc["pages"] == [1, 2]


def test_span_starting_in_the_middle_of_a_word_includes_that_word():
    doc = build_document(PAGES)
    start = doc["text"].index("Agreement") + 3
    loc = span_to_locations(doc, start, start + 2)
    assert loc["pages"] == [1] and len(loc["highlights"][0]["rects"]) == 1


def test_empty_and_out_of_range_spans():
    doc = build_document(PAGES)
    assert span_to_locations(doc, 5, 5)["pages"] == []
    assert span_to_locations(doc, 10_000, 10_010)["pages"] == []
    assert span_to_locations(build_document([[]]), 0, 5)["pages"] == []


def test_blank_pages_and_scan_detection():
    doc = build_document([PAGES[0], [], []])
    assert doc["n_pages"] == 3 and doc["pages"][1]["char_start"] is None
    assert doc["likely_scanned"] is True  # 3 words over 3 pages
    assert build_document([[w(0, 0, 1, 1, "x%d" % i) for i in range(40)]])["likely_scanned"] is False


def make_pdf(path, pages):
    pdf = pymupdf.open()
    for lines in pages:
        page = pdf.new_page()
        y = 72
        for line in lines:
            page.insert_text((72, y), line, fontsize=11)
            y += 16
    pdf.save(str(path))
    pdf.close()


def test_real_pdf_roundtrip(tmp_path):
    p = tmp_path / "c.pdf"
    make_pdf(p, [["This Agreement is made on January 5, 2020", "between Acme Corp and Beta LLC."],
                 ["This Agreement shall be governed by the laws of Delaware."]])
    doc = extract_pdf(p)
    assert doc["n_pages"] == 2 and len(doc["page_sizes"]) == 2
    assert "governed by the laws of Delaware." in doc["text"]
    start = doc["text"].index("governed by the laws of Delaware")
    loc = span_to_locations(doc, start, start + len("governed by the laws of Delaware"))
    assert loc["pages"] == [2]
    x0, y0, x1, y1 = loc["highlights"][0]["rects"][0]
    assert 0 < x0 < x1 < doc["page_sizes"][1][0] and 0 < y0 < y1 < doc["page_sizes"][1][1]


def test_non_pdf_raises_value_error(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_text("this is not a pdf")
    with pytest.raises(ValueError):
        extract_pdf(p)