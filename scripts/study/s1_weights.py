"""Study 1: weight-space comparison of acceleration adapters, with no image generation.

Per adapter: how concentrated each layer's change is and where the energy sits. Per pair:
whether the layers' strongest directions agree, against three references: random vectors,
the same pair with layers of equal shape mismatched, and style LoRAs.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from adapters import ACCEL, ADAPTERS, BLOCKS, KINDS, STYLE, block, kind, load_factors, unet_layer_names  # noqa: E402

from wspd import config  # noqa: E402

K = 4  # subspace size for the top-k overlap


def profile(f) -> dict:
    """Concentration and location of one adapter's energy."""
    energy, top1, rank90 = defaultdict(float), [], []
    for name, (_, s, _) in f.items():
        e = float((s**2).sum())
        for g in (kind(name), block(name), "all"):
            energy[g] += e
        energy["top1"] += float(s[0] ** 2)
        top1.append(float(s[0] ** 2) / e)
        rank90.append(int((torch.cumsum(s**2, 0) / e < 0.9).sum()) + 1)
    total = energy["all"]
    return {
        "layers": len(f),
        "rank": int(max(len(s) for _, s, _ in f.values())),
        "norm": total**0.5,
        "top1_energy_share": energy["top1"] / total,
        "top1_share_median_layer": float(np.median(top1)),
        "rank_for_90pct_energy_median": float(np.median(rank90)),
        "energy_by_block": {b: energy[b] / total for b in BLOCKS},
        "energy_by_kind": {k: energy[k] / total for k in KINDS},
    }


def layer_agreement(a, b) -> dict:
    """Agreement between two layers' factorized changes."""
    (ua, sa, va), (ub, sb, vb) = a, b
    out_cos = float(ua[:, 0] @ ub[:, 0])
    in_cos = float(va[0] @ vb[0])
    inner = float(((ua.T @ ub) * sa[:, None] * sb[None, :] * (va @ vb.T)).sum())  # <dA, dB> without forming either
    k = min(K, ua.shape[1], ub.shape[1])
    return {
        "out": abs(out_cos),
        "in": abs(in_cos),
        "signed_rank1": out_cos * in_cos,
        "delta_cos": inner / float(sa.norm() * sb.norm()),
        "inner": inner,
        "out_topk": float((ua[:, :k].T @ ub[:, :k]).pow(2).sum()) / k,
        "random_out": math.sqrt(2 / (math.pi * ua.shape[0])),
        "random_topk": k / ua.shape[0],
        "weight": float(sa.norm() * sb.norm()),
    }


def summarize(rows: list[dict]) -> dict:
    w = np.array([r["weight"] for r in rows])
    w = w / w.sum()
    col = lambda k: np.array([r[k] for r in rows])  # noqa: E731
    return {
        "layers": len(rows),
        "out_cos_weighted": float(col("out") @ w),
        "out_cos_median": float(np.median(col("out"))),
        "in_cos_weighted": float(col("in") @ w),
        "in_cos_median": float(np.median(col("in"))),
        "signed_rank1_weighted": float(col("signed_rank1") @ w),
        "same_sign_share": float((col("signed_rank1") > 0).mean()),
        "out_topk_weighted": float(col("out_topk") @ w),
        "delta_cos_global": float(col("inner").sum() / np.array([r["weight"] for r in rows]).sum()),
        "random_out_cos": float(col("random_out") @ w),
        "random_topk": float(col("random_topk") @ w),
    }


def compare(fa, fb) -> dict:
    shared = [n for n in fa if n in fb]
    rows = {n: layer_agreement(fa[n], fb[n]) for n in shared}
    result = summarize(list(rows.values()))
    # Control: pair each layer with a different layer of the same shape (the next one, cyclically).
    by_shape = defaultdict(list)
    for n in shared:
        by_shape[(fa[n][0].shape[0], fa[n][2].shape[1])].append(n)
    mismatched = [layer_agreement(fa[n], fb[group[(i + 1) % len(group)]]) for group in by_shape.values() if len(group) > 1 for i, n in enumerate(group)]
    result["mismatched_out_cos_weighted"] = summarize(mismatched)["out_cos_weighted"]
    result["mismatched_in_cos_weighted"] = summarize(mismatched)["in_cos_weighted"]
    for label, fn in (("by_block", block), ("by_kind", kind)):
        groups = defaultdict(list)
        for n, r in rows.items():
            groups[fn(n)].append(r)
        result[label] = {g: {k: v for k, v in summarize(rs).items() if k in ("layers", "out_cos_weighted", "in_cos_weighted", "delta_cos_global")} for g, rs in groups.items()}
    return result


def main() -> None:
    names = unet_layer_names()
    factors = {k: load_factors(k, names) for k in ADAPTERS}
    profiles = {k: profile(f) for k, f in factors.items()}
    for k, p in profiles.items():
        print(f"{k:14s} {p['layers']:3d} layers rank {p['rank']:3d} | top-1 energy {p['top1_energy_share']:.3f} | rank90 {p['rank_for_90pct_energy_median']:4.0f} | up {p['energy_by_block']['up_blocks']:.2f} | time {p['energy_by_kind']['time']:.3f} | norm {p['norm']:.1f}", flush=True)

    pairs = {}
    for a, b in combinations(ADAPTERS, 2):
        pairs[f"{a}|{b}"] = r = compare(factors[a], factors[b])
        print(f"{a:14s} {b:14s} out {r['out_cos_weighted']:.3f} (random {r['random_out_cos']:.3f}, mismatched {r['mismatched_out_cos_weighted']:.3f}) | in {r['in_cos_weighted']:.3f} | delta {r['delta_cos_global']:+.3f} | top{K} {r['out_topk_weighted']:.3f}", flush=True)

    out = config.RESULTS / "study"
    out.mkdir(parents=True, exist_ok=True)
    (out / "s1_weights.json").write_text(
        json.dumps({"adapters": {k: {"repo": a.repo, "file": a.file, "family": a.family} for k, a in ADAPTERS.items()}, "accel": ACCEL, "style": STYLE, "topk": K, "profiles": profiles, "pairs": pairs}, indent=1)
    )


if __name__ == "__main__":
    main()
