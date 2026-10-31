# DocuMind AI

Find a clause in a contract, see the exact passage highlighted on its page, and see how sure the model is. When a clause is missing, DocuMind simply says no.

DocuMind checks **41 clause types** from the CUAD dataset (governing law, cap on liability, anti-assignment, audit rights, non-compete and so on). It is built around three ideas:

- **Evidence, not summaries.** Every answer is the original passage, highlighted on the original page.
- **A score you can read.** Each answer shows its score against the decision threshold, and close calls are flagged.
- **It can say "not found".** The model is trained to recognise that a clause is absent, and a threshold chosen on validation data decides when to abstain.

Everything in the clause finder runs on your own computer. An optional "Ask a question" mode sends your question and a few retrieved excerpts to Groq, only after you tick a permission box each time.

## What is in the app

| Feature | What it does |
| --- | --- |
| Clause finder | Checks all 41 clause types in a PDF, ranks them by score, highlights the passage on the page. |
| Word and other formats | `.docx`, `.doc`, `.odt` and `.rtf` files are converted to PDF with LibreOffice, then analysed. |
| Scanned PDFs | PDFs with no text layer are read with Tesseract OCR first. Marked clearly as OCR text. |
| Ask a question | Works on any document. Hybrid retrieval (BM25 + dense) finds excerpts, Groq writes a short cited answer, or replies exactly "Not found in the document." |
| PDF report | A readable report: summary, every clause found with its passage and page, the clauses not found. |
| JSON export | The full analysis (scores, margins, passages, highlight positions) for other tools. |
| Interface | Light and dark theme, Back and Forward navigation, drag and drop, results kept in `storage\<id>\results.json`. |

## Results

All numbers below are on the **held-out test contracts**. The abstention threshold was chosen on the validation split only, and the test split was scored once. Nothing was tuned on it.

| Model | AUPR | Precision at 80% recall | F1 with abstention | Absent clauses reported as not found | Present clauses found | Right passage in top 3 (flagged) |
| --- | --- | --- | --- | --- | --- | --- |
| Legal-BERT (main) | 0.939 | 0.918 | 0.894 | 96.7% | 80.9% | 96.4% |
| bert-base-uncased | 0.918 (95% CI 0.904 to 0.930) | 0.887 | 0.882 | 96.5% | 76.8% | 96.8% |

- A model that always answers "not found" scores F1 0.703 on the same metric, so 0.894 is a real gain.
- Validation: Legal-BERT AUPR 0.944, F1 0.881, threshold 8.690. bert-base AUPR 0.927, F1 0.867, threshold 9.077.
- bert-base macro AUPR over 40 categories: 0.833.
- Legal-BERT was ahead of bert-base on every headline metric except the top-3 passage rate. I describe it as "consistently ahead", not "significantly ahead", because I have not run a paired significance test between the two.
- Metrics: AUPR (present vs absent), precision at 80% recall, token F1 with abstention, hit@k for passage retrieval, risk-coverage curves, bootstrap confidence intervals resampled by contract.
- Retrieval baselines (BM25, dense `bge-small`, hybrid reciprocal rank fusion), risk-coverage plots and per-category numbers are in the `reports\` folder.

### Speed and OCR (measured on a laptop with a 4 GB GPU)

| Measurement | Result |
| --- | --- |
| 5,381-word contract | 21.8 s with an idle GPU |
| 40-page, 12,998-word licence | 42 to 57 s (about 3 to 4 s per 1,000 words) |
| 30-page contract | 32 s |
| OCR on a printed 6-page research paper | 93.4% of words recovered, 8.0% extra garbled words, about 7.7 s per page |
| OCR on a 47-page handwritten notebook | 183 s in total, but the text is mostly garbled (handwriting is not supported) |

The OCR accuracy figure was measured on a research paper, not on a contract.

## How it works

1. **Text with positions.** PyMuPDF extracts every word with its page and bounding box, so any character span can be mapped back to highlight rectangles on the page.
2. **Question answering.** CUAD is regrouped by (contract, category). Each of the 41 clause types is a question asked of the contract. Long contracts are cut into overlapping windows (384 tokens, stride 128) in the SQuAD v2 style, with the CLS token as the "no answer" option.
3. **Present or absent.** `present_score = best span score - lowest null score`. A clause counts as found when the score is at or above the threshold chosen on validation data.
4. **Ask mode.** The document is cut into 120-word chunks, ranked by BM25 and by `bge-small` embeddings, and merged with reciprocal rank fusion. The top excerpts are sent to Groq with an instruction to cite them and to answer exactly "Not found in the document." when the excerpts do not contain the answer.
5. **Serving.** FastAPI with one worker thread (one GPU), a single-page web interface, and results written to `storage\`.

## Run it

Windows, Python 3.13 in the `.venv313` virtual environment, an NVIDIA GPU is recommended.

```
cd /d E:\Documind
call .venv313\Scripts\activate.bat
python -m uvicorn documind.api.main:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000.

- **Analyse a PDF from the command line:** `python -m documind.pipeline.analyze path\to\contract.pdf`
- **Run the tests (144):** `python -m pytest -q`. If your C: drive is short of space, add `--basetemp=E:\Documind\.pytest_tmp`.
- **Word files** need LibreOffice, and **scans** need Tesseract. Both are found automatically, or set `DOCUMIND_SOFFICE` and `DOCUMIND_TESSERACT` to their paths.
- **Ask mode** needs a Groq API key. Put one line, `GROQ_API_KEY=your-key`, in a file named `.env` in the project folder (it is git-ignored), or set the variable in your environment. The model and limits are in `configs\general.json`. `python -m scripts.groq_models` lists the model names your key can use.

## Project layout

| Folder | Contents |
| --- | --- |
| `documind\pdf` | Text and word-box extraction, highlight rendering, Word conversion and OCR |
| `documind\models`, `documind\eval` | Question-answering model, training, scoring, abstention and error analysis |
| `documind\retrieval` | BM25, dense retrieval and fusion baselines |
| `documind\pipeline` | The 41-clause analysis |
| `documind\general` | Ask mode (retrieval and Groq) |
| `documind\api`, `documind\web` | FastAPI backend and the single-page interface |
| `documind\report.py`, `documind\envfile.py` | PDF report and `.env` loader |
| `configs`, `scripts`, `tests`, `reports` | Settings, helper scripts, test suite, evaluation outputs |

## Limitations

- **It misses about 1 in 5 present clauses** (80.9% found). "Not found" is a strong hint, not proof: check the document.
- **Scores are raw model scores**, not calibrated probabilities. "Close call" means the margin over the threshold is under 3.
- **Handwriting is not supported**, and OCR on scans can contain recognition errors. OCR processes at most 100 pages.
- **English-language commercial contracts only.** The model is trained on CUAD, so unusual documents (job descriptions, lecture notes) will mostly return "not found".
- **Ask mode highlights the whole retrieved chunk** (about 120 words), not just the sentence the answer used, and answers are written by an AI model: check the highlighted page. Groq's free tier has rate limits, and models there are retired from time to time.
- **One document at a time.** The job list is kept in memory and is lost when the server restarts. Finished results stay in `storage\`.
- **PyMuPDF is AGPL-licensed.** Check its terms before redistributing the project.
- This is a research and portfolio project and **not legal advice**.

## Data

The model is trained and tested on [CUAD](https://www.atticusprojectai.org/cuad), the Contract Understanding Atticus Dataset: 510 commercial contracts annotated with 41 clause types, released under CC BY 4.0. The data is not included in this repository.