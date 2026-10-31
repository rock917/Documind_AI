from pathlib import Path

from documind.envfile import load_env_file

# Read secrets such as GROQ_API_KEY from E:\Documind\.env (if it exists) before anything else starts.
load_env_file(Path(__file__).resolve().parents[2] / ".env")

from documind.api.app import create_app
from documind.general.qa import GeneralMode
from documind.pipeline.analyze import ClauseFinder

app = create_app(ClauseFinder.from_settings("configs/inference.json"), "storage",
                 general=GeneralMode.from_settings("configs/general.json"))