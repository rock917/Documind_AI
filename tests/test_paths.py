from pathlib import Path

from documind.data.paths import windows_path


def test_default_tag_keeps_original_file_name():
    assert windows_path("data/processed", "train") == Path("data/processed/windows_train.jsonl")


def test_tag_is_added_before_extension():
    assert windows_path("data/processed", "val", "_bert") == Path("data/processed/windows_val_bert.jsonl")