"""Get a document ready for the reader: convert Word-style files to PDF (LibreOffice) and OCR scanned PDFs (Tesseract).

Both tools are external programs. They are looked up (in this order) from an environment variable,
the PATH, then the usual Windows install folders. A missing tool gives a plain ValueError message.
"""
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import pymupdf

OFFICE_EXTS = {".docx", ".doc", ".odt", ".rtf"}
SOFFICE_ENV, TESSERACT_ENV = "DOCUMIND_SOFFICE", "DOCUMIND_TESSERACT"
SOFFICE_WIN = [r"C:\Program Files\LibreOffice\program\soffice.exe", r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"]
TESSERACT_WIN = [r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]

_office_lock = threading.Lock()  # LibreOffice cannot run two conversions on one profile at once


def find_tool(name, env_var, windows_paths):
    """Path of an external program, or None."""
    override = os.environ.get(env_var)
    if override:
        return override if Path(override).is_file() else None
    found = shutil.which(name)
    if found:
        return found
    for p in windows_paths:
        if Path(p).is_file():
            return p
    return None


def find_soffice():
    return find_tool("soffice", SOFFICE_ENV, SOFFICE_WIN)


def find_tesseract():
    return find_tool("tesseract", TESSERACT_ENV, TESSERACT_WIN)


def office_to_pdf(src, out_dir, timeout=180):
    """Convert a Word/ODT/RTF file to a PDF in out_dir and return the PDF path."""
    soffice = find_soffice()
    if soffice is None:
        raise ValueError("Word files need LibreOffice, which was not found on this computer.")
    src, out_dir = Path(src), Path(out_dir)
    profile = Path(tempfile.gettempdir()) / "documind_lo_profile"
    cmd = [soffice, f"-env:UserInstallation={profile.resolve().as_uri()}", "--headless", "--convert-to", "pdf",
           "--outdir", str(out_dir), str(src)]
    with _office_lock:
        try:
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise ValueError("Converting the Word file took too long and was stopped.")
    pdf = out_dir / (src.stem + ".pdf")
    if run.returncode != 0 or not pdf.is_file():
        raise ValueError("This Word file could not be converted to PDF.")
    return pdf


def ocr_pdf(src_pdf, dst_pdf, dpi=200, lang="eng", progress=None, max_pages=100, timeout_per_page=180):
    """Write dst_pdf: every page of src_pdf as an image with an invisible text layer from Tesseract."""
    tesseract = find_tesseract()
    if tesseract is None:
        raise ValueError("Scanned files need Tesseract OCR, which was not found on this computer.")
    with pymupdf.open(str(src_pdf)) as src:
        n = len(src)
        if n > max_pages:
            raise ValueError(f"This scan has {n} pages; OCR is limited to {max_pages}.")
        out = pymupdf.open()
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(n):
                png = Path(tmp) / f"p{i}.png"
                src[i].get_pixmap(dpi=dpi).save(str(png))
                base = Path(tmp) / f"o{i}"
                try:
                    run = subprocess.run([tesseract, str(png), str(base), "-l", lang, "pdf"],
                                         capture_output=True, text=True, timeout=timeout_per_page)
                except subprocess.TimeoutExpired:
                    raise ValueError(f"OCR took too long on page {i + 1} and was stopped.")
                page_pdf = base.with_suffix(".pdf")
                if run.returncode != 0 or not page_pdf.is_file():
                    raise ValueError(f"OCR failed on page {i + 1}.")
                with pymupdf.open(str(page_pdf)) as one:
                    out.insert_pdf(one)
                if progress:
                    progress(i + 1, n)
        out.save(str(dst_pdf))
        out.close()
    return Path(dst_pdf)