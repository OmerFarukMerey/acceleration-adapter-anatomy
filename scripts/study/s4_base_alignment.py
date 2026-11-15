"""Study 4: are the adapters' top directions already special in the base model?

For each layer, the top output direction u and input direction v of an adapter are compared
with the singular vectors of the base weight. "Share in top 10%" is the fraction of the
direction that lies in the span of the strongest tenth of the base weight's singular vectors;
a direction unrelated to the base weight would give 0.10. No image generation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import UNet2DConditionModel

sys.path.insert(0, str(Path(__file__).parent))
from adapters import ADAPTERS, BASE, BASE_REV, load_factors  # noqa: E402

from wspd import config  # noqa: E402

KEYS = ["lcm", "tcd", "pcm", "hyper4", "flash", "style_monet_a", "style_pixel"]
FRACTION = 0.10


def main() -> None:
    state = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet", revision=BASE_REV).state_dict()
    factors = {k: load_factors(k) for k in KEYS}
    layers = sorted({n for f in factors.values() for n in f})
    rows = {k: [] for k in KEYS}
    for i, name in enumerate(layers):
        w = state[f"{name}.weight"].flatten(1).double()
        device = "mps" if torch.backends.mps.is_available() and False else "cpu"
        uw, sw, vwh = torch.linalg.svd(w.to(device), full_matrices=False)
        top = max(1, int(FRACTION * len(sw)))
        for k, f in factors.items():
            if name not in f:
                continue
            u, s, vh = f[name]
            cu, cv = (uw.T @ u[:, 0].double()), (vwh @ vh[0].double())
            rows[k].append({
                "energy": float((s**2).sum()),
                "out_share_top": float((cu[:top] ** 2).sum()),
                "in_share_top": float((cv[:top] ** 2).sum()),
                "in_share_in_row_space": float((cv**2).sum()),
                "out_max_cos": float(cu.abs().max()),
                "expected": top / len(sw),
            })
        if i % 40 == 0:
            print(f"{i}/{len(layers)} layers", flush=True)
    summary = {}
    for k, rs in rows.items():
        wgt = np.array([r["energy"] for r in rs]) ** 0.5
        wgt = wgt / wgt.sum()
        summary[k] = {c: float(np.array([r[c] for r in rs]) @ wgt) for c in ("out_share_top", "in_share_top", "in_share_in_row_space", "out_max_cos", "expected")}
        print(f"{k:14s} " + " | ".join(f"{c} {v:.3f}" for c, v in summary[k].items()), flush=True)
    out = config.RESULTS / "study"
    out.mkdir(parents=True, exist_ok=True)
    (out / "s4_base_alignment.json").write_text(json.dumps({"fraction": FRACTION, "adapters": {k: ADAPTERS[k].repo for k in KEYS}, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
