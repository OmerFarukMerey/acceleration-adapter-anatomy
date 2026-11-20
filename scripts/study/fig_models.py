"""Figure: agreement between independent methods on four base models, against the controls.

From s1_weights.json (Stable Diffusion 1.5) and s8_other_models.json (SDXL, SD3 Medium, FLUX.1-dev).
"""
from __future__ import annotations

import json
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wspd import config

# One adapter per independent method, per base model.
INDEPENDENT = {
    "Stable Diffusion 1.5": ["lcm", "tcd", "pcm", "hyper4", "flash"],
    "SDXL": ["lcm", "tcd", "pcm", "hyper4", "dmd2", "lightning4", "flash", "tdd"],
    "SD3 Medium": ["hyper4", "pcm4", "flash"],
    "FLUX.1-dev": ["hyper8", "turbo", "tdd"],
}
BLUE, GRAY, INK, MUTED = "#256abf", "#8a8a84", "#1f1f1c", "#6b6b66"


def main() -> None:
    out = config.RESULTS / "study"
    other = json.loads((out / "s8_other_models.json").read_text())
    sources = {"Stable Diffusion 1.5": json.loads((out / "s1_weights.json").read_text())["pairs"], "SDXL": other["sdxl"]["pairs"], "SD3 Medium": other["sd3"]["pairs"], "FLUX.1-dev": other["flux"]["pairs"]}
    fig, ax = plt.subplots(figsize=(9.6, 3.9), dpi=160)
    rng = np.random.default_rng(0)
    summary = {}
    for row, (model, keys) in enumerate(INDEPENDENT.items()):
        pairs = sources[model]
        get = lambda a, b: pairs.get(f"{a}|{b}") or pairs[f"{b}|{a}"]  # noqa: E731
        agree = [get(a, b)["out_cos_weighted"] for a, b in combinations(keys, 2)]
        control = [get(a, b)["mismatched_out_cos_weighted"] for a, b in combinations(keys, 2)]
        control += [p["out_cos_weighted"] for k, p in pairs.items() if ("style" in k.split("|")[0]) != ("style" in k.split("|")[1])]
        y = len(INDEPENDENT) - 1 - row
        ax.scatter(control, y - 0.16 + rng.uniform(-0.07, 0.07, len(control)), s=22, color=GRAY, edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter(agree, y + 0.16 + rng.uniform(-0.07, 0.07, len(agree)), s=34, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(0.83, y + 0.16, f"{len(keys)} methods, {len(agree)} pairs: {min(agree):.2f} to {max(agree):.2f}", ha="left", va="center", fontsize=8.5, color=INK, clip_on=False)
        ax.text(0.83, y - 0.16, f"controls: at most {max(control):.2f}", ha="left", va="center", fontsize=8.5, color=MUTED, clip_on=False)
        summary[model] = {"methods": len(keys), "pairs": len(agree), "min": min(agree), "max": max(agree), "mean": float(np.mean(agree)), "control_max": max(control)}
    ax.set_yticks(range(len(INDEPENDENT)), list(INDEPENDENT)[::-1], fontsize=10, color=INK)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-0.55, len(INDEPENDENT) - 0.45)
    ax.set_xlabel("Agreement of the top output direction per layer (absolute cosine, energy-weighted)", fontsize=9, color=MUTED)
    ax.grid(axis="x", color="#e6e6e1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9, labelcolor=MUTED)
    ax.tick_params(axis="y", labelcolor=INK, labelsize=10)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.scatter([], [], s=34, color=BLUE, label="pairs of independent few-step methods")
    ax.scatter([], [], s=22, color=GRAY, label="controls: mismatched layers, and few-step against style adapters")
    fig.legend(loc="upper left", bbox_to_anchor=(0.005, 0.93), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK, handletextpad=0.3, columnspacing=1.5)
    fig.suptitle("Independent few-step methods agree on a weight direction on every base model tested", x=0.012, ha="left", fontsize=11.5, color=INK, y=0.985)
    fig.tight_layout(rect=(0, 0, 0.76, 0.86))
    fig.savefig(out / "s8_models.png", facecolor="white")
    (out / "s8_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
