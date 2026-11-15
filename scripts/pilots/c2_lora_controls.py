"""Pilot C2: is the rank-4 result about structure, or would any few directions do?

Controls for pilot C on the same adapter, prompts and seeds. Keeping the top singular
directions of each module is compared with keeping random ones, with keeping only the
ones truncation throws away, and with a decoder-only version.
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from c_lora_anatomy import LORA, N_PROMPTS, STEPS, line, load, score, sheet  # noqa: E402

from wspd import config  # noqa: E402
from wspd.metrics import Evaluator  # noqa: E402
from wspd.store import device_lock  # noqa: E402


def band(lo: int, hi: int):
    return lambda name, n: torch.arange(lo, min(hi, n))


def random(k: int, seed: int):
    def pick(name, n):
        gen = torch.Generator().manual_seed(zlib.crc32(name.encode()) + seed)
        return torch.randperm(n, generator=gen)[:k]

    return pick


everywhere = lambda n: True  # noqa: E731
decoder = lambda n: n.startswith("up_blocks")  # noqa: E731
ARMS = {
    "none": (lambda n: False, band(0, 0)),
    "full": (everywhere, band(0, 64)),
    "rank_4": (everywhere, band(0, 4)),
    "rank_2": (everywhere, band(0, 2)),
    "rank_1": (everywhere, band(0, 1)),
    "next_4 (5-8)": (everywhere, band(4, 8)),
    "tail (5-64)": (everywhere, band(4, 64)),
    "random_4_a": (everywhere, random(4, 0)),
    "random_4_b": (everywhere, random(4, 1)),
    "decoder_rank_4": (decoder, band(0, 4)),
    "decoder_rank_1": (decoder, band(0, 1)),
}


def main() -> None:
    bench = load()
    parts = bench.parts
    total_energy = sum(float((s**2).sum()) for _, s, _ in parts.values())
    total_params = sum((u.shape[0] + vh.shape[1]) * len(s) for u, s, vh in parts.values())
    size, images = {}, {}
    for name, (keep, pick) in ARMS.items():
        kept = {n: pick(n, len(parts[n][1])) for n in parts if keep(n)}
        size[name] = {
            "energy_share": sum(float((parts[n][1][i] ** 2).sum()) for n, i in kept.items()) / total_energy,
            "lora_param_share": sum((parts[n][0].shape[0] + parts[n][2].shape[1]) * len(i) for n, i in kept.items()) / total_params,
        }
        bench.install(keep, pick=pick)
        images[name] = bench.generate()
        print(f"generated {name}", flush=True)
    bench.install(lambda n: False)

    results = score(Evaluator(bench.device), images, bench.prompts)
    for name, r in results.items():
        r.update(size[name])
        print(f"{line(name, r)} | energy {r['energy_share']:.3f} | params {r['lora_param_share']:.3f}", flush=True)

    out = config.RESULTS / "pilots"
    show = ["none", "random_4_a", "tail (5-64)", "rank_1", "rank_2", "decoder_rank_4", "rank_4", "full"]
    sheet(images, show, out / "c2_lora_controls.png")
    (out / "c2_lora_controls.json").write_text(
        json.dumps({"lora": LORA, "steps": STEPS, "n_prompts": N_PROMPTS, "sheet_columns": show, "arms": results}, indent=1)
    )


if __name__ == "__main__":
    with device_lock():
        main()
