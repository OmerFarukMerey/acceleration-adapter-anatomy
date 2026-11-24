"""Image sheets for studies 16 to 18, from the images those studies wrote under runs/study/.

    python scripts/study/fig_sheets.py
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from wspd import config

STUDY = config.RUNS / "study"
OUT = config.RESULTS / "study"
SIZE = 144


def tile(folder, index: int) -> np.ndarray:
    return np.asarray(Image.open(folder / f"{index:05d}.png").convert("RGB").resize((SIZE, SIZE), Image.LANCZOS))


def sheet(root, rows: list[list[str]], captions: list[int], path, across: bool = True) -> None:
    """One block per caption, side by side or (across=False) one under the other. Each row of a block
    lists the arms shown left to right."""
    blocks = [np.concatenate([np.concatenate([tile(root / arm, index) for arm in row], 1) for row in rows], 0) for index in captions]
    gap = np.full((blocks[0].shape[0], 8, 3) if across else (8, blocks[0].shape[1], 3), 255, np.uint8)
    parts = [x for block in blocks for x in (block, gap)][:-1]
    Image.fromarray(np.concatenate(parts, 1 if across else 0)).save(path, quality=88)
    print(path)


def main() -> None:
    captions = [0, 3]
    if (STUDY / "quality_steps").exists():
        other = {1: 8, 2: 8, 4: 1, 8: 1}  # the step count whose strength is borrowed in the last-but-one column
        rows = [[f"hyper{k}_{a}" for a in ("none", "full", "rank1", "shared", "profile", f"str{other[k]}", "rand")] for k in (1, 2, 4, 8)]
        sheet(STUDY / "quality_steps", rows, captions, OUT / "s16_steps.jpg")
    if (STUDY / "quality_kinds").exists():
        rows = [["none_lcm", "flash_full", "flash_rank1", "flash_const", "flash_const_values", "flash_const2_values", "flash_const2", "flash_posavg", "flash_posrec"]]
        sheet(STUDY / "quality_kinds", rows, [0, 3, 5], OUT / "s17_flash.jpg", across=False)
    for tag in ("ds8", "rv51"):
        root = STUDY / f"quality_ft_{tag}"
        if root.exists():
            rows = [[f"none_{s}", f"{h}_full", f"{h}_rank1", f"{h}_const", f"{h}_loo"] for h, s in (("lcm", "lcm"), ("tcd", "tcd"), ("hyper4", "ddim"))]
            rows.append(["ref_base25", "base4_guided", "none_ddim", "trained_strength", "trained_gradient"])
            sheet(root, rows, captions, OUT / f"s18_finetune_{tag}.jpg")


if __name__ == "__main__":
    main()
