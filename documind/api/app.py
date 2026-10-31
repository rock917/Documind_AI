"""DocuMind web API: upload a PDF, watch progress, fetch the clause results and page images with highlights.

One background worker runs the analyses one after another (the model lives on a single GPU).
Endpoints:
    POST /api/documents                       upload a PDF (multipart field "file") -> document info, analysis starts
    GET  /api/documents/{id}                  state (queued | running | done | error) and progress
    GET  /api/documents/{id}/results          the clause results once the state is done
    GET  /api/documents/{id}/page/{n}.png     page n as an image; ?hl=START-END (repeatable) highlights character spans
    GET  /                                    the web page (documind/web/index.html) when it exists
"""
import json
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path

import pymupdf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from documind.pdf.extract import extract_pdf, span_to_locations
from documind.pdf.prepare import OFFICE_EXTS, find_tesseract, ocr_pdf, office_to_pdf
from documind.report import build_report

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
HIGHLIGHT_RGB = (1.0, 0.85, 0.0)


def render_page(pdf_path, page_no, rects, dpi=110):
    """PNG bytes of one page with the given boxes (page coordinates) painted as translucent highlights."""
    with pymupdf.open(str(pdf_path)) as pdf:
        page = pdf[page_no - 1]
        for r in rects:
            page.draw_rect(pymupdf.Rect(*r), color=None, fill=HIGHLIGHT_RGB, fill_opacity=0.4, overlay=True)
        return page.get_pixmap(dpi=dpi).tobytes("png")


def parse_spans(values):
    spans = []
    for v in values:
        try:
            a, b = v.split("-")
            a, b = int(a), int(b)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Bad highlight {v!r}; use START-END with whole numbers.")
        if a < 0 or b <= a:
            raise HTTPException(status_code=400, detail=f"Bad highlight {v!r}; need 0 <= START < END.")
        spans.append((a, b))
    return spans


class AskBody(BaseModel):
    question: str
    allow_network: bool = False


class JobManager:
    def __init__(self, finder, storage_dir):
        self.finder = finder
        self.root = Path(storage_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, data, filename, analyze=True):
        doc_id = uuid.uuid4().hex
        folder = self.root / doc_id
        folder.mkdir(parents=True)
        pdf_path = folder / "original.pdf"
        try:
            if Path(filename).suffix.lower() in OFFICE_EXTS:
                source = folder / ("source" + Path(filename).suffix.lower())
                source.write_bytes(data)
                office_to_pdf(source, folder).replace(pdf_path)
            else:
                pdf_path.write_bytes(data)
            doc = extract_pdf(pdf_path)
            needs_ocr = doc["n_words"] == 0
            if needs_ocr and find_tesseract() is None:
                raise ValueError("This PDF has no extractable text (it looks like a scan), and Tesseract OCR is not installed.")
        except ValueError:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        rec = {"id": doc_id, "filename": filename, "pdf_path": pdf_path, "doc": doc, "state": "queued",
               "done": 0, "total": len(self.finder.questions), "current": None, "error": None,
               "results": None, "seconds": None, "needs_ocr": needs_ocr, "ocr": False, "analyze": analyze, "retriever": None}
        self.jobs[doc_id] = rec
        self.queue.put(doc_id)
        return self.public(rec)

    @staticmethod
    def public(rec):
        d = rec["doc"]
        return {"id": rec["id"], "filename": rec["filename"], "n_pages": d["n_pages"], "n_words": d["n_words"],
                "state": rec["state"], "done": rec["done"], "total": rec["total"], "current": rec["current"],
                "error": rec["error"], "seconds": rec["seconds"], "page_sizes": d["page_sizes"], "ocr": rec.get("ocr", False), "analyze": rec.get("analyze", True)}

    def _worker(self):
        while True:
            doc_id = self.queue.get()
            rec = self.jobs[doc_id]
            rec["state"] = "running"
            started = time.time()

            def progress(done, total, category):
                rec["done"], rec["total"], rec["current"] = done, total, category

            try:
                if rec["needs_ocr"]:
                    rec["state"] = "ocr"
                    rec["done"], rec["total"] = 0, rec["doc"]["n_pages"]
                    ocr_path = rec["pdf_path"].parent / "ocr.pdf"
                    ocr_pdf(rec["pdf_path"], ocr_path, progress=lambda d, t: rec.update(done=d, total=t))
                    ocr_doc = extract_pdf(ocr_path)
                    if ocr_doc["n_words"] == 0:
                        raise ValueError("OCR could not read any text in this file.")
                    rec["pdf_path"], rec["doc"], rec["ocr"] = ocr_path, ocr_doc, True
                    rec["done"], rec["total"], rec["state"] = 0, len(self.finder.questions), "running"
                results = self.finder.analyze(rec["doc"], None, progress) if rec["analyze"] else []
                rec["results"] = results
                rec["seconds"] = round(time.time() - started, 1)
                payload = {"document": self.public(rec), "threshold": self.finder.threshold, "results": results}
                (self.root / doc_id / "results.json").write_text(json.dumps(payload), encoding="utf-8")
                rec["state"] = "done"
            except Exception as exc:  # report any failure to the client instead of killing the worker
                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["state"] = "error"


def create_app(finder, storage_dir="storage", web_dir=None, general=None):
    app = FastAPI(title="DocuMind AI")
    jobs = JobManager(finder, storage_dir)
    web_index = Path(web_dir or Path(__file__).resolve().parent.parent / "web") / "index.html"

    def get_rec(doc_id):
        rec = jobs.jobs.get(doc_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="Unknown document id.")
        return rec

    @app.get("/")
    def home():
        if web_index.exists():
            return FileResponse(web_index)
        return JSONResponse({"message": "DocuMind API is running. The web page is not installed yet."})

    @app.post("/api/documents")
    async def upload(file: UploadFile = File(...), analyze: bool = True):
        name = file.filename or "document.pdf"
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File is larger than 50 MB.")
        ext = Path(name).suffix.lower()
        if ext in OFFICE_EXTS:
            if not data:
                raise HTTPException(status_code=400, detail="This file is empty.")
        elif ext != ".pdf" or not data.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="Please upload a PDF or Word file (.pdf, .docx, .doc, .odt, .rtf).")
        try:
            return jobs.submit(data, name, analyze)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/api/documents/{doc_id}")
    def status(doc_id: str):
        return jobs.public(get_rec(doc_id))

    @app.get("/api/documents/{doc_id}/results")
    def results(doc_id: str):
        rec = get_rec(doc_id)
        if rec["state"] == "error":
            raise HTTPException(status_code=500, detail=rec["error"])
        if rec["state"] != "done":
            raise HTTPException(status_code=409, detail=f"Analysis is {rec['state']}; try again shortly.")
        return {"document": jobs.public(rec), "threshold": finder.threshold, "results": rec["results"]}

    @app.get("/api/documents/{doc_id}/page/{page_no}.png")
    def page_image(doc_id: str, page_no: int, hl: list[str] = Query(default=[]), dpi: int = Query(110, ge=50, le=200)):
        rec = get_rec(doc_id)
        if not 1 <= page_no <= rec["doc"]["n_pages"]:
            raise HTTPException(status_code=404, detail="No such page.")
        rects = []
        for start, end in parse_spans(hl):
            for h in span_to_locations(rec["doc"], start, end)["highlights"]:
                if h["page"] == page_no:
                    rects.extend(h["rects"])
        return Response(content=render_page(rec["pdf_path"], page_no, rects, dpi), media_type="image/png")

    @app.get("/api/documents/{doc_id}/report.pdf")
    def report(doc_id: str):
        rec = get_rec(doc_id)
        if rec["state"] == "error":
            raise HTTPException(status_code=500, detail=rec["error"])
        if rec["state"] != "done":
            raise HTTPException(status_code=409, detail=f"Analysis is {rec['state']}; try again shortly.")
        if not rec.get("analyze", True):
            raise HTTPException(status_code=409, detail="This document was opened in Ask mode, so there is no clause analysis to report.")
        pdf = build_report(jobs.public(rec), finder.threshold, rec["results"])
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(rec["filename"]).stem)[:60] or "document"
        return Response(content=pdf, media_type="application/pdf", headers={"Content-Disposition": f'inline; filename="{stem}_report.pdf"'})

    @app.get("/api/general")
    def general_info():
        if general is None:
            return {"available": False, "model": None, "key_variable": "GROQ_API_KEY"}
        return general.info()

    @app.get("/api/about")
    def about():
        return {"categories": list(getattr(finder, "questions", {})), "threshold": getattr(finder, "threshold", None)}

    @app.post("/api/documents/{doc_id}/ask")
    def ask(doc_id: str, body: AskBody):
        rec = get_rec(doc_id)
        if general is None or not general.available():
            raise HTTPException(status_code=503, detail="General mode is off. Start the server with the GROQ_API_KEY environment variable set.")
        if not body.allow_network:
            raise HTTPException(status_code=400, detail="General mode sends your question and excerpts of the document to Groq. Tick the box to allow it.")
        question = body.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Please type a question.")
        limit = general.cfg["max_question_chars"]
        if len(question) > limit:
            raise HTTPException(status_code=400, detail=f"Questions are limited to {limit} characters.")
        if not rec["doc"]["n_words"]:
            raise HTTPException(status_code=409, detail="The document is still being prepared; try again shortly.")
        try:
            return general.ask(rec, question)
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    return app
