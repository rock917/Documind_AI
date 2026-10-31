import json

import numpy as np
import torch

from documind.models.train_qa import WindowDataset, collate


def write_windows(path, n=10):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            length = 5 + i
            f.write(json.dumps({
                "input_ids": list(range(1, length + 1)),
                "token_type_ids": [0] * 3 + [1] * (length - 3),
                "start_position": i % 3,
                "end_position": i % 3,
            }) + "\n")


def test_dataset_loads_rows_as_compact_arrays(tmp_path):
    p = tmp_path / "w.jsonl"
    write_windows(p)
    ds = WindowDataset(p)
    assert len(ds) == 10
    item = ds[0]
    assert item["input_ids"].dtype == np.int32 and item["token_type_ids"].dtype == np.int8
    assert item["start_positions"] == 0 and item["end_positions"] == 0


def test_limit_takes_a_deterministic_subset(tmp_path):
    p = tmp_path / "w.jsonl"
    write_windows(p, n=50)
    a = WindowDataset(p, limit=7, seed=3)
    b = WindowDataset(p, limit=7, seed=3)
    assert len(a) == 7
    assert [len(r[0]) for r in a.rows] == [len(r[0]) for r in b.rows]


def test_collate_pads_to_longest_and_builds_mask(tmp_path):
    p = tmp_path / "w.jsonl"
    write_windows(p)
    ds = WindowDataset(p)
    batch = collate([ds[0], ds[9]], pad_id=0)  # lengths 5 and 14
    assert batch["input_ids"].shape == (2, 14)
    assert batch["attention_mask"][0].sum().item() == 5
    assert batch["attention_mask"][1].sum().item() == 14
    assert batch["input_ids"][0, 5:].eq(0).all()
    assert batch["start_positions"].dtype == torch.long