"""Study 9: fully fine-tuned few-step models, as a check that the findings are not LoRA artifacts.

A LoRA can only make a low-rank change. A fully fine-tuned student has no such limit, so its
weight change (student minus its own base model) shows whether one direction per layer is what
few-step training wants, and whether that direction is the one the LoRA adapters found.

For every layer the SD 1.5 adapters touch: the spectrum of the full change, and the agreement
of its top direction with each adapter's top direction. No image generation.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent))
from adapters import ACCEL, BASE, BASE_REV, STYLE, load_factors, unet_layer_names  # noqa: E402
from s1_weights import layer_agreement, summarize  # noqa: E402

from wspd import config  # noqa: E402

KEEP = 64
SD15 = (BASE, BASE_REV, "unet/diffusion_pytorch_model.safetensors")


@dataclass(frozen=True)
class Full:
    repo: str
    rev: str
    file: str
    base: tuple[str, str, str]
    method: str


FULL = {
    "dmd2_full": Full("tianweiy/DMD2", "be22767697a1f3ca656b73c776e15fa335c86c6c", "model/sdv1.5/laion6.25_sd_baseline_8node_guidance1.75_lr5e-7_seed10_dfake10_diffusion1000_gan1e-3_resume_fid8.35_checkpoint_model_041000/pytorch_model.bin", SD15, "distribution matching + adversarial, 1 step"),
    "reflow2_full": Full("XCLiu/2_rectified_flow_from_sd_1_5", "934962eb1054b67883fabb3e0860b851baf31142", "unet/diffusion_pytorch_model.safetensors", SD15, "rectified flow (2-reflow)"),
    "lcm_full": Full("SimianLuo/LCM_Dreamshaper_v7", "a85df6a8bd976cdd08b4fd8f3b73f229c9e54df5", "unet/diffusion_pytorch_model.safetensors", ("Lykon/dreamshaper-7", "9b481047f77996efa025e75e03941dbf51f506ad", "unet/diffusion_pytorch_model.safetensors"), "consistency, on the DreamShaper 7 base"),
}


def weights(repo: str, rev: str, file: str) -> dict[str, torch.Tensor]:
    path = hf_hub_download(repo, file, revision=rev)
    state = load_file(path) if path.endswith(".safetensors") else torch.load(path, map_location="cpu", weights_only=True)
    return {k.removeprefix("module.").removeprefix("unet."): v for k, v in state.items()}


def factors(key: str, layers: list[str]):
    """Top directions and the whole spectrum of student minus base, per layer."""
    cache = config.RUNS / "study" / f"factors_{key}.pt"
    if cache.exists():
        return torch.load(cache)
    m = FULL[key]
    student, base = weights(m.repo, m.rev, m.file), weights(*m.base)
    out = {}
    for n in layers:
        delta = (student[f"{n}.weight"].double() - base[f"{n}.weight"].double()).flatten(1)
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        out[n] = (u[:, :KEEP].float(), s[:KEEP].float(), vh[:KEEP].float(), s.float(), float(base[f"{n}.weight"].double().norm()))
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return out


def profile(f) -> dict:
    top1, rank90, rel, e_top, e_all = [], [], [], 0.0, 0.0
    for u, s, vh, spectrum, base_norm in f.values():
        e = float((spectrum**2).sum())
        top1.append(float(spectrum[0] ** 2) / e)
        rank90.append((int((torch.cumsum(spectrum**2, 0) / e < 0.9).sum()) + 1) / len(spectrum))
        rel.append(e**0.5 / base_norm)
        e_top += float(spectrum[0] ** 2)
        e_all += e
    return {"layers": len(f), "top1_energy_share": e_top / e_all, "top1_share_median_layer": float(np.median(top1)), "share_of_rank_for_90pct_energy_median": float(np.median(rank90)), "relative_change_median": float(np.median(rel))}


def main() -> None:
    names = unet_layer_names()
    loras = {k: load_factors(k, names) for k in [*ACCEL, *STYLE]}
    layers = sorted(loras["lcm"])
    out = config.RESULTS / "study" / "s9_full_models.json"
    results = json.loads(out.read_text()) if out.exists() else {"profiles": {}, "pairs": {}}
    full = {}
    for key in sys.argv[1:] or list(FULL):
        full[key] = f = factors(key, layers)
        results["profiles"][key] = p = profile(f)
        print(f"{key:13s} top-1 energy {p['top1_energy_share']:.3f} (median layer {p['top1_share_median_layer']:.3f}) | 90% energy needs {p['share_of_rank_for_90pct_energy_median']:.2f} of the rank | relative change {p['relative_change_median']:.4f}", flush=True)
        thin = {n: v[:3] for n, v in f.items()}
        for other, g in loras.items():
            shared = [n for n in thin if n in g]
            r = summarize([layer_agreement(thin[n], g[n]) for n in shared])
            # How much of the full change lies along the adapter's top direction, against its own top direction.
            along = sum(float((thin[n][0].T @ g[n][0][:, 0] * thin[n][1] * (thin[n][2] @ g[n][2][0])).sum()) ** 2 for n in shared)
            r["energy_share_along_adapter_top_direction"] = along / sum(float((f[n][3] ** 2).sum()) for n in shared)
            results["pairs"][f"{key}|{other}"] = r
            print(f"{key:13s} {other:14s} out {r['out_cos_weighted']:.3f} (random {r['random_out_cos']:.3f}) | in {r['in_cos_weighted']:.3f} | delta {r['delta_cos_global']:+.3f} | same sign {r['same_sign_share']:.2f} | energy along adapter direction {r['energy_share_along_adapter_top_direction']:.3f}", flush=True)
    for a, b in combinations(full, 2):
        ta, tb = {n: v[:3] for n, v in full[a].items()}, {n: v[:3] for n, v in full[b].items()}
        results["pairs"][f"{a}|{b}"] = r = summarize([layer_agreement(ta[n], tb[n]) for n in ta])
        print(f"{a:13s} {b:14s} out {r['out_cos_weighted']:.3f} (random {r['random_out_cos']:.3f}) | in {r['in_cos_weighted']:.3f} | delta {r['delta_cos_global']:+.3f} | same sign {r['same_sign_share']:.2f}", flush=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
