"""Analyse one public contract once, on this computer, and save it as a sample for the web app.

Run from the project folder with the virtual environment active:
    python -m scripts.make_sample path\\to\\contract.pdf --name cuad-sample
Writes samples\\<name>\\original.pdf and samples\\<name>\\results.json. The web app lists every folder in samples\\
as a ready-made example that opens instantly.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

from documind.pdf.extract import extract_pdf
from documind.pipeline.analyze import ClauseFinder

def main():
    p = argparse.ArgumentParser()
    p.add_argument("pdf")
    p.add_argument("--name", default="sample", help="folder name under samples\\ (default: sample)")
    p.add_argument("--settings", default="configs/inference.json")
    args = p.parse_args()

    src = Path(args.pdf)
    if not src.is_file():
        raise SystemExit(f"STOP: {src} not found.")
    doc = extract_pdf(src)
    if doc["n_words"] == 0:
        raise SystemExit("STOP: this PDF has no extractable text.")
    finder = ClauseFinder.from_settings(args.settings)
    print(f"{src.name}: {doc['n_pages']} pages, {doc['n_words']} words | device: {finder.device}")
    start = time.time()
    results = finder.analyze(doc, None, lambda done, total, cat: print(f"  {done}/{total} {cat}", end="\r"))
    seconds = round(time.time() - start, 1)

    out = Path("samples") / args.name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out / "original.pdf")
    payload = {"filename": src.name, "seconds": seconds, "threshold": finder.threshold, "results": results}
    (out / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    found = sum(1 for r in results if r["present"])
    print(f"\nanalysed in {seconds} s: {found} of {len(results)} clause types found")
    print(f"saved {out / 'original.pdf'} and {out / 'results.json'}")

if __name__ == "__main__":
    main()