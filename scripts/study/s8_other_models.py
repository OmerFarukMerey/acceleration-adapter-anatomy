"""Study 8: the weight-space comparison on other base models, with no image generation.

Repeats study 1 (concentration per layer, agreement of the top directions between adapters)
and study 5 (shared starting point) for the few-step adapters of SDXL, Stable Diffusion 3
Medium and FLUX.1-dev. Only the adapter files are read; no base model is loaded.

Files that store no LoRA alpha (the transformer adapters) are read at scale 1. That leaves
directions and energy shares right and makes the reported norms a convention.

    python scripts/study/s8_other_models.py [sdxl sd3 flux]
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent))
from adapters import factorize  # noqa: E402
from s1_weights import layer_agreement, summarize  # noqa: E402

from wspd import config  # noqa: E402

SKIP = 8  # directions left out of the shared-starting-point test


@dataclass(frozen=True)
class File:
    repo: str
    file: str
    rev: str
    method: str  # adapters with the same method are one release family


_H = ("ByteDance/Hyper-SD", "bc08d970a87c74c71209491d64e3525845698863")
_P = ("wangfuyun/PCM_Weights", "39560fead4ce00f94db3cb8e93dd8fba90ec0be6")
_T = ("RED-AIGC/TDD", "285176b159c322c04c97b7aaf43a0a9505f303af")
_L = ("ByteDance/SDXL-Lightning", "c9a24f48e1c025556787b0c58dd67a091ece2e44")
MODELS: dict[str, dict[str, File]] = {
    "sdxl": {
        "lcm": File("latent-consistency/lcm-lora-sdxl", "pytorch_lora_weights.safetensors", "a18548dd4956b174ec5b0d78d340c8dae0a129cd", "LCM"),
        "tcd": File("h1t/TCD-SDXL-LoRA", "pytorch_lora_weights.safetensors", "81499fdd85015a3d54595bbdddb5fe29d449e103", "TCD"),
        "pcm": File(*_P[:1], "sdxl/pcm_sdxl_smallcfg_4step_converted.safetensors", _P[1], "PCM"),
        "pcm_lcmlike": File(*_P[:1], "sdxl/pcm_sdxl_lcmlike_lora_converted.safetensors", _P[1], "PCM"),
        "hyper4": File(*_H[:1], "Hyper-SDXL-4steps-lora.safetensors", _H[1], "Hyper-SD"),
        "hyper1": File(*_H[:1], "Hyper-SDXL-1step-lora.safetensors", _H[1], "Hyper-SD"),
        "hyper2": File(*_H[:1], "Hyper-SDXL-2steps-lora.safetensors", _H[1], "Hyper-SD"),
        "hyper8": File(*_H[:1], "Hyper-SDXL-8steps-lora.safetensors", _H[1], "Hyper-SD"),
        "dmd2": File("tianweiy/DMD2", "dmd2_sdxl_4step_lora_fp16.safetensors", "be22767697a1f3ca656b73c776e15fa335c86c6c", "DMD2"),
        "lightning4": File(*_L[:1], "sdxl_lightning_4step_lora.safetensors", _L[1], "SDXL-Lightning"),
        "lightning2": File(*_L[:1], "sdxl_lightning_2step_lora.safetensors", _L[1], "SDXL-Lightning"),
        "lightning8": File(*_L[:1], "sdxl_lightning_8step_lora.safetensors", _L[1], "SDXL-Lightning"),
        "flash": File("jasperai/flash-sdxl", "pytorch_lora_weights.safetensors", "84e93604226d43e74ae9085a98ea4eaee57d8600", "Flash Diffusion"),
        "tdd": File(*_T[:1], "sdxl_tdd_lora_weights.safetensors", _T[1], "TDD"),
        "tdd_wo_adv": File(*_T[:1], "sdxl_tdd_wo_adv_lora.safetensors", _T[1], "TDD"),
        "style_pixel": File("nerijs/pixel-art-xl", "pixel-art-xl.safetensors", "8bf4a4d9ea283e00a51fafda8e0539f8248ea037", "style"),
        "style_papercut": File("TheLastBen/Papercut_SDXL", "papercut.safetensors", "b001f28d3913bc1e4a19f2564fa41e39049bec1c", "style"),
        "style_crayon": File("ostris/crayon_style_lora_sdxl", "crayons_v1_sdxl.safetensors", "aef7fd10d197d2a80b008e9e8f0575da0a312025", "style"),
    },
    "sd3": {
        "hyper4": File(*_H[:1], "Hyper-SD3-4steps-CFG-lora.safetensors", _H[1], "Hyper-SD"),
        "hyper8": File(*_H[:1], "Hyper-SD3-8steps-CFG-lora.safetensors", _H[1], "Hyper-SD"),
        "pcm4": File(*_P[:1], "sd3/pcm_deterministic_4step_shift3.safetensors", _P[1], "PCM"),
        "pcm2": File(*_P[:1], "sd3/pcm_deterministic_2step_shift1.safetensors", _P[1], "PCM"),
        "flash": File("jasperai/flash-sd3", "adapter_model.safetensors", "fb184d916f9193a81a4264758eb0ebbff1c56b69", "Flash Diffusion"),
    },
    "flux": {
        "hyper8": File(*_H[:1], "Hyper-FLUX.1-dev-8steps-lora.safetensors", _H[1], "Hyper-SD"),
        "hyper16": File(*_H[:1], "Hyper-FLUX.1-dev-16steps-lora.safetensors", _H[1], "Hyper-SD"),
        "turbo": File("alimama-creative/FLUX.1-Turbo-Alpha", "diffusion_pytorch_model.safetensors", "7b29bf041b413949089d78b826a5c9726e3c8e50", "Turbo-Alpha"),
        "tdd": File(*_T[:1], "FLUX.1-dev_tdd_lora_weights.safetensors", _T[1], "TDD"),
        "tdd_adv": File(*_T[:1], "FLUX.1-dev_tdd_adv_lora_weights.safetensors", _T[1], "TDD"),
    },
}

# SDXL layer names in the original (SGM) layout, mapped to the diffusers layout.
_SGM = {**{f"input_blocks_{i}_1": f"down_blocks_{1 + (i > 6)}_attentions_{(i - 4) % 3}" for i in (4, 5, 7, 8)}, "middle_block_1": "mid_block_attentions_0", **{f"output_blocks_{i}_1": f"up_blocks_{i // 3}_attentions_{i % 3}" for i in range(6)}}
_PATTERNS = (
    (r"lora_unet_(.+)\.lora_down\.weight", "{}.lora_up.weight", "{}.alpha"),
    (r"unet\.(.+)\.lora\.down\.weight", "{}.lora.up.weight", None),
    (r"(?:transformer|base_model\.model)\.(.+)\.lora_A\.weight", "{}.lora_B.weight", None),
)


def layer_id(stem: str) -> str:
    flat = stem.replace(".", "_")
    for sgm, diffusers in _SGM.items():
        if flat.startswith(sgm + "_"):
            return diffusers + flat[len(sgm) :]
    return flat


def load(model: str, key: str):
    cache = config.RUNS / "study" / f"factors_{model}_{key}.pt"
    if cache.exists():
        return {n: f for n, f in torch.load(cache).items() if float(f[1][0]) > 0}  # some files hold untrained, all-zero layers
    a = MODELS[model][key]
    raw = load_file(hf_hub_download(a.repo, a.file, revision=a.rev))
    out = {}
    for k in sorted(raw):
        for pattern, up_name, alpha_name in _PATTERNS:
            m = re.fullmatch(pattern, k)
            if not m:
                continue
            prefix = k[: m.start(1)] + m.group(1)
            down, up = raw[k].float(), raw[up_name.format(prefix)].float()
            alpha = raw.get(alpha_name.format(prefix)) if alpha_name else None
            scale = float(alpha) / down.shape[0] if alpha is not None else 1.0
            out[layer_id(m.group(1))] = factorize(up, down, scale)
    assert out, f"no LoRA layers recognised in {a.file}"
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return {n: f for n, f in out.items() if float(f[1][0]) > 0}


def kind(name: str) -> str:
    for label, pattern in (("time", r"time_emb|time_text_embed|timestep"), ("self_attn", r"attn1_"), ("cross_attn", r"attn2_"), ("attn", r"attn_"), ("ff", r"_ff_|ff_context|proj_mlp"), ("norm", r"norm"), ("conv", r"conv|resnets|samplers")):
        if re.search(pattern, name):
            return label
    return "other"


def profile(f) -> dict:
    energy, top1 = defaultdict(float), []
    for name, (_, s, _) in f.items():
        e = float((s**2).sum())
        energy[kind(name)] += e
        energy["all"] += e
        energy["top1"] += float(s[0] ** 2)
        top1.append(float(s[0] ** 2) / e)
    total = energy["all"]
    return {
        "layers": len(f),
        "rank": int(max(len(s) for _, s, _ in f.values())),
        "top1_energy_share": energy["top1"] / total,
        "top1_share_median_layer": float(np.median(top1)),
        "energy_by_kind": {k: v / total for k, v in energy.items() if k not in ("all", "top1")},
    }


def compare(fa, fb) -> dict | None:
    shared = [n for n in fa if n in fb and fa[n][0].shape[0] == fb[n][0].shape[0] and fa[n][2].shape[1] == fb[n][2].shape[1]]
    if not shared:
        return None
    result = summarize([layer_agreement(fa[n], fb[n]) for n in shared])
    by_shape = defaultdict(list)
    for n in shared:
        by_shape[(fa[n][0].shape[0], fa[n][2].shape[1])].append(n)
    mismatched = [layer_agreement(fa[n], fb[g[(i + 1) % len(g)]]) for g in by_shape.values() if len(g) > 1 for i, n in enumerate(g)]
    result["mismatched_out_cos_weighted"] = summarize(mismatched)["out_cos_weighted"]
    # Shared starting point: overlap of the weak input directions, against unrelated subspaces.
    tail, unrelated = [], []
    for n in shared:
        va, vb = fa[n][2][SKIP:], fb[n][2][SKIP:]
        if len(va) and len(vb):
            tail.append(float((va @ vb.T).pow(2).sum()) / min(len(va), len(vb)))
            unrelated.append(max(len(va), len(vb)) / va.shape[1])
    result["input_tail_overlap"], result["input_tail_unrelated"] = float(np.mean(tail)), float(np.mean(unrelated))
    return result


def main() -> None:
    out = config.RESULTS / "study" / "s8_other_models.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    for model in sys.argv[1:] or list(MODELS):
        factors = {k: load(model, k) for k in MODELS[model]}
        profiles = {k: profile(f) for k, f in factors.items()}
        for k, p in profiles.items():
            print(f"{model:5s} {k:15s} {p['layers']:4d} layers rank {p['rank']:3d} | top-1 energy {p['top1_energy_share']:.3f} | median layer {p['top1_share_median_layer']:.3f}", flush=True)
        pairs = {}
        for a, b in combinations(factors, 2):
            r = compare(factors[a], factors[b])
            if r is None:
                continue
            pairs[f"{a}|{b}"] = r
            print(f"{model:5s} {a:15s} {b:15s} {r['layers']:4d} layers | out {r['out_cos_weighted']:.3f} (random {r['random_out_cos']:.3f}, mismatched {r['mismatched_out_cos_weighted']:.3f}) | in {r['in_cos_weighted']:.3f} | delta {r['delta_cos_global']:+.3f} | same sign {r['same_sign_share']:.2f} | start overlap {r['input_tail_overlap']:.3f} (unrelated {r['input_tail_unrelated']:.3f})", flush=True)
        results[model] = {"adapters": {k: {"repo": a.repo, "file": a.file, "rev": a.rev, "method": a.method} for k, a in MODELS[model].items()}, "profiles": profiles, "pairs": pairs}
        out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
