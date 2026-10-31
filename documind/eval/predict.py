"""Run a trained QA model over ALL windows of a split and save one prediction per (contract, category).

No sampling and no labels are used here: windows are built with an empty answer list, so nothing
about the gold answers can leak into the predictions. Gold answers are only copied into the output
so the evaluation script can compare.

Run from the project root (E:\\Documind) with the virtual environment active, for example:
    python -m documind.eval.predict --model-dir checkpoints\\legalbert_qa\\final --name legalbert_qa --split val
Writes data\\predictions\\<name>_<split>.jsonl
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from documind.data.windows import make_windows
from documind.eval.qa_decode import aggregate_windows, decode_window


def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def window_logits(model, windows, batch_size=16, device="cpu", pad_id=0):
    """Start/end logits for each window, in order, each trimmed to the window's real length."""
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            chunk = windows[i:i + batch_size]
            n = max(len(w["input_ids"]) for w in chunk)
            ids = torch.full((len(chunk), n), pad_id, dtype=torch.long)
            types = torch.zeros((len(chunk), n), dtype=torch.long)
            mask = torch.zeros((len(chunk), n), dtype=torch.long)
            for j, w in enumerate(chunk):
                length = len(w["input_ids"])
                ids[j, :length] = torch.tensor(w["input_ids"], dtype=torch.long)
                types[j, :length] = torch.tensor(w["token_type_ids"], dtype=torch.long)
                mask[j, :length] = 1
            res = model(input_ids=ids.to(device), token_type_ids=types.to(device), attention_mask=mask.to(device))
            starts = res.start_logits.float().cpu().numpy()
            ends = res.end_logits.float().cpu().numpy()
            for j, w in enumerate(chunk):
                length = len(w["input_ids"])
                out.append((starts[j, :length], ends[j, :length]))
    return out


def finite_or_none(x):
    return float(x) if x is not None and math.isfinite(x) else None


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--name", required=True, help="label used in the output file name")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--tokenizer", default="nlpaueb/legal-bert-base-uncased")
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--top-k", type=int, default=5, help="how many non-overlapping candidate spans to save per example")
    p.add_argument("--limit-examples", type=int, default=0, help="random subset for a quick trial (0 = all)")
    args = p.parse_args()

    processed = Path(cfg["data"]["processed_dir"])
    contracts = json.loads((processed / "contracts.json").read_text(encoding="utf-8"))
    questions = json.loads((processed / "questions.json").read_text(encoding="utf-8"))
    examples = read_jsonl(processed / f"{args.split}.jsonl")
    if args.limit_examples and args.limit_examples < len(examples):
        examples = random.Random(cfg["seed"]).sample(examples, args.limit_examples)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir)
    pad_id = model.config.pad_token_id or 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model.half()
    model.to(device)
    print(f"device: {device} | examples: {len(examples)} | split: {args.split} | model: {args.model_dir}")

    out_dir = Path(cfg["data"]["processed_dir"]).parent / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}_{args.split}.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        for e in tqdm(examples, desc=f"predict[{args.split}]"):
            ctx = contracts[e["title"]]
            wins = make_windows(tok, questions[e["category"]], ctx, [], args.max_length, args.stride)
            logits = window_logits(model, wins, args.batch_size, device, pad_id)
            decoded = [decode_window(s, en, w["offset_mapping"], top_spans=args.top_k) for (s, en), w in zip(logits, wins)]
            agg = aggregate_windows(decoded, ctx, k=args.top_k)
            out.write(json.dumps({
                "title": e["title"],
                "category": e["category"],
                "answers": e["answers"],
                "has_answer": bool(e["answers"]),
                "pred_text": agg["text"],
                "char_start": agg["char_start"],
                "char_end": agg["char_end"],
                "span_score": finite_or_none(agg["span_score"]),
                "null_score": finite_or_none(agg["null_score"]),
                "present_score": finite_or_none(agg["present_score"]),
                "n_windows": len(wins),
                "candidates": [{"text": c["text"], "char_start": c["char_start"], "char_end": c["char_end"],
                                "span_score": finite_or_none(c["span_score"]),
                                "present_score": finite_or_none(c["present_score"])} for c in agg["candidates"]],
            }, ensure_ascii=False) + "\n")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()