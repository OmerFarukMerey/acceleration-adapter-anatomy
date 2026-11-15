"""Figure: pairwise agreement of the adapters' top output directions, from s1_weights.json."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from wspd import config

LABELS = {
    "lcm": "LCM", "tcd": "TCD", "pcm": "PCM", "pcm_lcmlike": "PCM (LCM-like)", "flash": "Flash",
    "hyper1": "Hyper-SD 1-step", "hyper2": "Hyper-SD 2-step", "hyper4": "Hyper-SD 4-step", "hyper8": "Hyper-SD 8-step",
    "style_monet_a": "Style: Monet A", "style_monet_b": "Style: Monet B", "style_pixel": "Style: pixel art",
}
ORDER = list(LABELS)
BLUES = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED = "#1f1f1c", "#6b6b66"


def main() -> None:
    out = config.RESULTS / "study"
    pairs = json.loads((out / "s1_weights.json").read_text())["pairs"]
    n = len(ORDER)
    grid = np.full((n, n), np.nan)
    for i, a in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            if i != j:
                grid[i, j] = (pairs.get(f"{a}|{b}") or pairs[f"{b}|{a}"])["out_cos_weighted"]
    fig, ax = plt.subplots(figsize=(9.2, 7.6), dpi=160)
    image = ax.imshow(grid, cmap=LinearSegmentedColormap.from_list("blue", BLUES), vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if grid[i, j] > 0.55 else INK)
    names = [LABELS[k] for k in ORDER]
    ax.set_xticks(range(n), names, rotation=45, ha="right", fontsize=9, color=INK)
    ax.set_yticks(range(n), names, fontsize=9, color=INK)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for edge in (8.5,):  # separate acceleration adapters from the style controls
        ax.axhline(edge, color=MUTED, linewidth=1)
        ax.axvline(edge, color=MUTED, linewidth=1)
    ax.set_title("Agreement of the top output direction per layer between adapters", loc="left", fontsize=12, color=INK, pad=26)
    ax.text(0, 1.02, "Absolute cosine, averaged over shared layers and weighted by energy. Two random directions give about 0.02.", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(out / "s1_agreement.png", facecolor="white")


if __name__ == "__main__":
    main()
