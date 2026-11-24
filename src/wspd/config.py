"""Paths and constants shared by the scripts. Anything that affects results lives here."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ.get("WSP_RUNS_DIR", ROOT / "runs"))
RESULTS = ROOT / "results"
PROMPT_POOL = ROOT / "data" / "prompts_pool.json"

CLIP_ID = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

# The development prompts are a fixed split of the prompt pool: 3 prompts from each of 8
# categories. Changing any of these numbers changes which prompts the studies use.
SPLIT_SEED = 20261106
N_SEEN_CATEGORIES = 8
DEV_PER_CATEGORY = 3
