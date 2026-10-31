"""File-name helpers shared by the window builder and the trainer."""
from pathlib import Path


def windows_path(processed_dir, split, tag=""):
    """Path of the sampled-windows file. The empty tag keeps the original name (windows_train.jsonl);
    a tag such as '_bert' gives windows_train_bert.jsonl, so a second tokenizer never overwrites the first."""
    return Path(processed_dir) / f"windows_{split}{tag}.jsonl"