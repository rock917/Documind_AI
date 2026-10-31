"""Tiny .env loader: reads KEY=VALUE lines into os.environ. Never prints values; never overrides variables that are already set."""
import os
from pathlib import Path


def load_env_file(path):
    """Load `path` if it exists. Returns the names it set (never the values)."""
    p = Path(path)
    if not p.is_file():
        return []
    names = []
    for raw in p.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and value and key not in os.environ:
            os.environ[key] = value
            names.append(key)
    return names