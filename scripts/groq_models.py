"""List the model ids your Groq key can use, so you can pick a valid one for configs/general.json.

Reads GROQ_API_KEY from the environment or from the project's .env file. Prints model ids only, never the key.
Run from the project folder:  python -m scripts.groq_models
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from documind.envfile import load_env_file


def main():
    load_env_file(Path(".env"))
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        sys.exit("GROQ_API_KEY is not set (not in the environment and not in .env).")
    cfg = json.loads(Path("configs/general.json").read_text(encoding="utf-8"))
    req = urllib.request.Request(cfg["base_url"].rstrip("/") + "/models", headers={"Authorization": "Bearer " + key, "User-Agent": "DocuMind/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        sys.exit(f"Groq returned status {exc.code}." + (" The key was rejected." if exc.code in (401, 403) else ""))
    except (urllib.error.URLError, TimeoutError, OSError):
        sys.exit("Could not reach Groq. Check the internet connection.")
    ids = sorted(m.get("id", "") for m in data.get("data", []))
    print(f"{len(ids)} models available to this key. Configured model: {cfg['model']}")
    for i in ids:
        print(("* " if i == cfg["model"] else "  ") + i)
    if cfg["model"] not in ids:
        print("\nThe configured model is NOT in this list. Put one of the ids above into configs/general.json.")


main()
