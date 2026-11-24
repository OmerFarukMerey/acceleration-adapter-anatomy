"""Study 16: do Hyper-SD's adapters for 1, 2, 4 and 8 steps differ in direction, or only in strength?

The four adapters come from one method and one starting point, and their top directions agree
at 0.85 to 0.99 (study 1). Here one direction per layer is built from all four, and each adapter
is reduced to its strength along it: one number per layer and step count.

Weights (no image generation), into results/study/s16_steps.json:
  how much of each adapter the shared direction keeps, and whether the strengths of the four
  step counts are one profile over layers times one number per step count.

Images, for the first COCO captions of study 7 with the same seeds. Each step count k is sampled
with its own adapter's sampler and k steps, without guidance. Arms:
  hyper<k>_none      the base model
  hyper<k>_full      the adapter as released
  hyper<k>_rank1     its own top direction per layer
  hyper<k>_shared    the shared direction at this adapter's strength
  hyper<k>_loo       the direction of the other three step counts at this adapter's strength
  hyper<k>_profile   the shared direction at one profile over layers times one number for k
  hyper<k>_str<j>    the shared direction at the strength of the adapter for j steps
  hyper<k>_rand      a random direction per layer at this adapter's top strength

    python scripts/study/s16_steps.py [n_captions] [block]     # n_captions 0: weights only
    python scripts/study/s7_score.py --no-fid --steps
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from adapters import ADAPTERS, bench, block, factorize, load_factors, rank1  # noqa: E402
from s2_function import random_direction  # noqa: E402
from s7_quality import coco_items  # noqa: E402
from s12_ablation import sample  # noqa: E402

from wspd import config  # noqa: E402
from wspd.store import device_lock  # noqa: E402

COUNTS = (1, 2, 4, 8)
ROOT = config.RUNS / "study" / "quality_steps"


def direction(fs: list) -> dict:
    """Per layer, the top direction of the sum of the adapters' unit rank-1 changes."""
    out = {}
    for n in fs[0]:
        u, _, vh = factorize(torch.cat([f[n][0][:, :1] for f in fs], 1), torch.cat([f[n][2][:1] for f in fs], 0), 1.0)
        out[n] = (u[:, :1], vh[:1])
    return out


def along(f: dict, d: dict) -> dict:
    """The adapter's whole change measured along a direction: u' (U S V') v, per layer."""
    return {n: float(((d[n][0].T @ u) * s) @ (vh @ d[n][1].T)) for n, (u, s, vh) in f.items()}


def at(d: dict, strength: dict) -> dict:
    return {n: (u, torch.tensor([float(strength[n])], dtype=torch.float32), vh) for n, (u, vh) in d.items()}


def analysis(factors: dict) -> tuple[dict, dict]:
    """Returns the weight-space numbers and the arms' weight changes per step count."""
    shared = direction([factors[k] for k in COUNTS])
    layers = sorted(shared)
    energy = {k: sum(float(s.pow(2).sum()) for _, s, _ in factors[k].values()) for k in COUNTS}
    strength = {k: along(factors[k], shared) for k in COUNTS}
    a = np.array([[strength[k][n] for k in COUNTS] for n in layers])  # layers x step counts

    # Is the table of strengths one profile over layers times one number per step count?
    u, sv, vt = np.linalg.svd(a, full_matrices=False)
    ref = COUNTS.index(4)
    profile, number = u[:, 0] * sv[0] * vt[0, ref], vt[0] / vt[0, ref]
    fitted = np.outer(profile, number)

    result = {"layers": len(layers), "one_profile_share_of_strength_energy": float(sv[0] ** 2 / (sv**2).sum()), "per_step_count": {}}
    parts = {}
    for j, k in enumerate(COUNTS):
        f = factors[k]
        others = direction([factors[o] for o in COUNTS if o != k])
        loo = along(f, others)
        sq = lambda values, keep=lambda n: True: sum(v**2 for n, v in values.items() if keep(n))  # noqa: E731
        result["per_step_count"][str(k)] = {
            "sampler": ADAPTERS[f"hyper{k}"].sampler,
            "energy_in_own_top_direction": sum(float(s[0] ** 2) for _, s, _ in f.values()) / energy[k],
            "energy_along_shared_direction": sq(strength[k]) / energy[k],
            "energy_along_direction_of_the_other_three": sq(loo) / energy[k],
            "strength_size_relative_to_4_steps": float(np.linalg.norm(a[:, j]) / np.linalg.norm(a[:, ref])),
            "strength_correlation_with_4_steps": float(np.corrcoef(a[:, j], a[:, ref])[0, 1]),
            "same_sign_as_4_steps": float((np.sign(a[:, j]) == np.sign(a[:, ref])).mean()),
            "number_in_the_one_profile_fit": float(number[j]),
            "strength_energy_missed_by_the_fit": float(((a[:, j] - fitted[:, j]) ** 2).sum() / (a[:, j] ** 2).sum()),
            "strength_energy_by_block": {blk: sq(strength[k], lambda n, blk=blk: block(n) == blk) / sq(strength[k]) for blk in ("down_blocks", "mid_block", "up_blocks")},
        }
        parts[k] = {
            "none": None,
            "full": f,
            "rank1": rank1(f),
            "shared": at(shared, strength[k]),
            "loo": at(others, loo),
            "profile": at(shared, dict(zip(layers, fitted[:, j]))),
            "rand": random_direction(f),
            **{f"str{o}": at(shared, strength[o]) for o in COUNTS if o != k},
        }
    result["strength_correlation_between_step_counts"] = {f"{COUNTS[i]}|{COUNTS[j]}": float(np.corrcoef(a[:, i], a[:, j])[0, 1]) for i in range(len(COUNTS)) for j in range(i + 1, len(COUNTS))}
    return result, parts


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    result, parts = analysis({k: load_factors(f"hyper{k}") for k in COUNTS})
    (config.RESULTS / "study" / "s16_steps.json").write_text(json.dumps(result, indent=1))
    print(f"one profile times one number per step count holds {result['one_profile_share_of_strength_energy']:.3f} of the strengths' energy", flush=True)
    for k, r in result["per_step_count"].items():
        print(f"{k} steps | own top {r['energy_in_own_top_direction']:.3f} | shared {r['energy_along_shared_direction']:.3f} | other three {r['energy_along_direction_of_the_other_three']:.3f} | size vs 4 {r['strength_size_relative_to_4_steps']:.2f} | corr with 4 {r['strength_correlation_with_4_steps']:.2f} | number {r['number_in_the_one_profile_fit']:.2f} | missed {r['strength_energy_missed_by_the_fit']:.3f}", flush=True)
    if n == 0:
        return

    items = coco_items(n)
    b = bench()
    for start in range(0, n, size):
        for k in COUNTS:
            sampler = ADAPTERS[f"hyper{k}"].sampler
            for arm, change in parts[k].items():
                folder = ROOT / f"hyper{k}_{arm}"
                folder.mkdir(parents=True, exist_ok=True)
                todo = [(i, it) for i, it in enumerate(items[start : start + size], start) if not (folder / f"{i:05d}.png").exists()]
                if not todo:
                    continue
                t0 = time.time()
                b.install(change)
                for i, it in todo:
                    Image.fromarray(sample(b, it, sampler, k)).save(folder / f"{i:05d}.png")
                print(f"block {start:4d} | hyper{k}_{arm:8s} | {len(todo)} images | {time.time() - t0:4.0f} s", flush=True)
    b.install(None)


if __name__ == "__main__":
    with device_lock():
        main()
