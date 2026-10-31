"""Build a readable PDF report of one clause analysis (no network, no extra dependencies: PyMuPDF only)."""
import datetime
import html
import io

import pymupdf

CLOSE_MARGIN = 3          # same rule as the web page's "Close call" badge
MAX_PASSAGE_CHARS = 700

CSS = """
body { font-family: sans-serif; font-size: 10pt; color: #1b2430; }
h1 { font-size: 22pt; margin: 0 0 2pt 0; color: #26407a; }
h2 { font-size: 13pt; margin: 16pt 0 5pt 0; color: #26407a; border-bottom: 1px solid #c9d1de; }
p { margin: 0 0 5pt 0; }
.meta { color: #596579; font-size: 9pt; }
.clause { margin: 0 0 9pt 0; padding: 4pt 0 0 8pt; border-left: 3px solid #3556d6; }
.name { font-size: 11pt; font-weight: bold; }
.ok { color: #1f8a5b; font-weight: bold; }
.warn { color: #a15c07; font-weight: bold; }
.quote { font-family: serif; font-size: 10pt; margin: 3pt 0 2pt 0; }
.small { font-size: 8.5pt; color: #596579; }
.none { color: #596579; }
"""


def _fmt(x, digits=1, sign=False):
    if x is None:
        return "n/a"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def _pages(passage):
    pages = sorted(set(passage.get("pages") or []))
    return ", ".join(str(p) for p in pages) if pages else "page unknown"


def build_html(info, threshold, results):
    found = sorted((r for r in results if r.get("present")), key=lambda r: -(r.get("present_score") or 0))
    missing = sorted((r for r in results if not r.get("present")), key=lambda r: r.get("category", ""))
    esc = html.escape
    parts = ['<h1>DocuMind AI report</h1>']
    ocr = " Text was read with OCR, so it may contain recognition errors." if info.get("ocr") else ""
    parts.append(f'<p class="meta">{esc(str(info.get("filename", "document")))} &#183; {info.get("n_pages", "?")} pages &#183; '
                 f'generated {datetime.date.today().isoformat()}.{ocr}</p>')
    parts.append(f'<h2>Summary</h2><p><b>{len(found)} of {len(results)}</b> clause types were found. '
                 f'A clause counts as found at a score of {_fmt(threshold, 2)} or higher. '
                 f'The score is how much better the best passage beats "no such clause here". The further past the threshold, the more sure the model is.</p>')
    parts.append(f'<h2>Clauses found ({len(found)})</h2>')
    if not found:
        parts.append('<p class="none">No clause type reached the threshold.</p>')
    for r in found:
        margin = r.get("margin")
        close = margin is not None and margin < CLOSE_MARGIN
        flag = ' &#183; <span class="warn">Close call</span>' if close else ""
        passages = r.get("passages") or []
        block = [f'<div class="clause"><p class="name">{esc(r["category"])}</p>',
                 f'<p class="small"><span class="ok">Found</span> &#183; score {_fmt(r.get("present_score"))} '
                 f'(threshold {_fmt(threshold, 2)}, margin {_fmt(margin, 1, sign=True)}){flag}</p>']
        if passages:
            best = passages[0]
            text = " ".join(str(best.get("text", "")).split())
            if len(text) > MAX_PASSAGE_CHARS:
                text = text[:MAX_PASSAGE_CHARS].rstrip() + "..."
            block.append(f'<p class="quote">&#8220;{esc(text)}&#8221;</p>')
            block.append(f'<p class="small">Page {esc(_pages(best))}</p>')
        block.append("</div>")
        parts.append("".join(block))
    parts.append(f'<h2>Not found ({len(missing)})</h2>')
    if missing:
        parts.append('<p class="none">' + "; ".join(f'{esc(r["category"])} ({_fmt(r.get("present_score"))})' for r in missing) + ".</p>")
        parts.append('<p class="small">The number in brackets is the best score the model found for that clause type. It is below the threshold, so DocuMind reports the clause as absent. '
                     'A missing clause is a strong hint, not proof: always check the document.</p>')
    else:
        parts.append('<p class="none">Every clause type was found.</p>')
    parts.append('<h2>About this report</h2><p class="small">Produced by DocuMind AI using a Legal-BERT model fine-tuned on the CUAD dataset. '
                 'Scores are raw model scores compared with a threshold; they are not probabilities. This is a research and portfolio project and not legal advice.</p>')
    return "".join(parts)


def build_report(info, threshold, results):
    """Return the report as PDF bytes."""
    story = pymupdf.Story(html=build_html(info, threshold, results), user_css=CSS)
    buf = io.BytesIO()
    writer = pymupdf.DocumentWriter(buf)
    media = pymupdf.paper_rect("a4")
    where = media + (50, 50, -50, -56)
    more = 1
    while more:
        dev = writer.begin_page(media)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    doc = pymupdf.open(stream=buf.getvalue(), filetype="pdf")
    total = doc.page_count
    for i, page in enumerate(doc, start=1):
        page.insert_text((50, media.height - 30), f"DocuMind AI  |  page {i} of {total}", fontsize=8, color=(0.35, 0.4, 0.47))
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return out
