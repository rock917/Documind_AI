"""Fine-tune a BERT-style model for extractive QA with abstention on the sampled windows.

Input : data\\processed\\windows_train.jsonl and windows_val.jsonl (from build_windows).
Target: negative windows point at the CLS token (position 0), so the model can say "no answer".

Run from the project root (E:\\Documind) with the virtual environment active. Quick trial:
    python -m documind.models.train_qa --name trial --train-limit 800 --val-limit 400 --epochs 1 --batch-size 8 --grad-accum 1 --gradient-checkpointing
Writes reports\\train_<name>.json. With --save it also writes checkpoints\\<name>\\ (large files).
"""
import argparse
import functools
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from transformers import AutoModelForQuestionAnswering, Trainer, TrainingArguments

from documind.data.paths import windows_path


def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class WindowDataset(Dataset):
    """Windows from a JSONL file, kept as compact numpy arrays. `limit` takes a seeded random subset."""

    def __init__(self, path, limit=0, seed=42):
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.append((
                    np.asarray(r["input_ids"], dtype=np.int32),
                    np.asarray(r["token_type_ids"], dtype=np.int8),
                    int(r["start_position"]),
                    int(r["end_position"]),
                ))
        self.total_in_file = len(rows)
        if limit and limit < len(rows):
            rows = random.Random(seed).sample(rows, limit)
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, types, start, end = self.rows[i]
        return {"input_ids": ids, "token_type_ids": types, "start_positions": start, "end_positions": end}


def collate(batch, pad_id=0):
    """Pad a list of windows to the longest one in the batch."""
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    types = torch.zeros((len(batch), n), dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        length = len(b["input_ids"])
        ids[i, :length] = torch.from_numpy(b["input_ids"].astype(np.int64))
        types[i, :length] = torch.from_numpy(b["token_type_ids"].astype(np.int64))
        mask[i, :length] = 1
    return {
        "input_ids": ids,
        "token_type_ids": types,
        "attention_mask": mask,
        "start_positions": torch.tensor([b["start_positions"] for b in batch], dtype=torch.long),
        "end_positions": torch.tensor([b["end_positions"] for b in batch], dtype=torch.long),
    }


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="nlpaueb/legal-bert-base-uncased")
    p.add_argument("--name", default="legalbert_qa")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--val-limit", type=int, default=0)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--windows-tag", default="", help="which windows files to train on, e.g. _bert")
    p.add_argument("--save", action="store_true", help="save checkpoints each epoch and the final model")
    args = p.parse_args()

    processed = Path(cfg["data"]["processed_dir"])
    seed = cfg["seed"]
    train_ds = WindowDataset(windows_path(processed, "train", args.windows_tag), args.train_limit, seed)
    val_ds = WindowDataset(windows_path(processed, "val", args.windows_tag), args.val_limit, seed)

    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    pad_id = model.config.pad_token_id or 0
    use_cuda = torch.cuda.is_available()
    print(f"device: {'cuda' if use_cuda else 'cpu'} | train windows: {len(train_ds)} | val windows: {len(val_ds)}")

    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    total_steps = max(1, math.ceil(steps_per_epoch * args.epochs))
    out_dir = Path(cfg["paths"]["checkpoints_dir"]) / args.name

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=int(0.1 * total_steps),
        fp16=use_cuda,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="epoch",
        save_strategy="epoch" if args.save else "no",
        save_total_limit=1,
        logging_steps=args.logging_steps,
        label_names=["start_positions", "end_positions"],
        dataloader_num_workers=0,
        report_to="none",
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=functools.partial(collate, pad_id=pad_id),
    )

    start = time.time()
    result = trainer.train()
    wall = time.time() - start
    evals = [h for h in trainer.state.log_history if "eval_loss" in h]
    peak_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 2) if use_cuda else None

    summary = {
        "name": args.name,
        "model": args.model,
        "args": vars(args),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "optimizer_steps": result.global_step,
        "train_runtime_s": round(result.metrics.get("train_runtime", 0.0), 1),
        "train_windows_per_s": round(result.metrics.get("train_samples_per_second", 0.0), 2),
        "final_train_loss": result.metrics.get("train_loss"),
        "eval_loss": evals[-1]["eval_loss"] if evals else None,
        "peak_gpu_memory_gb": peak_gb,
        "wall_time_s": round(wall, 1),
        "log_history": trainer.state.log_history,
    }
    print("\n=== training summary ===")
    for key in ("train_windows", "val_windows", "optimizer_steps", "train_runtime_s", "train_windows_per_s",
                "final_train_loss", "eval_loss", "peak_gpu_memory_gb", "wall_time_s"):
        print(f"{key}: {summary[key]}")
    if args.train_limit and summary["train_windows_per_s"]:
        full = train_ds.total_in_file
        hours = full / summary["train_windows_per_s"] / 3600 * args.epochs
        print(f"rough estimate for all {full} train windows x {args.epochs} epochs at this speed: {hours:.1f} hours")

    rep = Path(cfg["paths"]["reports_dir"]) / f"train_{args.name}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if args.save:
        trainer.save_model(str(out_dir / "final"))
    print(f"\nSaved {rep}" + (f" and {out_dir / 'final'}" if args.save else ""))


if __name__ == "__main__":
    main()