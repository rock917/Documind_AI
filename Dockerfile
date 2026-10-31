FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libreoffice-writer libgomp1 fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

RUN python -c "import json; p='configs/inference.json'; c=json.load(open(p)); c['model_dir']='TheRock45/documind-legalbert-qa'; json.dump(c, open(p, 'w'), indent=2)"

RUN python -c "from huggingface_hub import snapshot_download; from transformers import AutoTokenizer; snapshot_download('TheRock45/documind-legalbert-qa'); AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')"

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "documind.api.main:app", "--host", "0.0.0.0", "--port", "7860"]