"""Study 11: do rank 1 and the constant per layer also hold in function on SDXL?

Study 8 compared SDXL adapters in weight space only. Here four of them are run: one image per
COCO caption at 1024 px, 4 steps, no guidance, saved at 512 px under runs/study/quality_sdxl/.
Arms:
  none_<sampler>   base SDXL with each adapter's sampler
  <host>_full      the adapter as released
  <host>_rank1     only its top direction per layer
  <host>_const     one fixed vector per layer added to the layer's output, the same at every step
The fixed vectors are estimated on the 16 development prompts, with the rank-1 adapter running.

    python scripts/study/s11_sdxl.py [n_captions] [block]
"""
from __future__ import annotations

import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, LCMScheduler, StableDiffusionXLPipeline, TCDScheduler
from huggingface_hub import hf_hub_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from adapters import factorize, rank1  # noqa: E402
from s3_mechanism import coefficient, shaped  # noqa: E402
from s7_quality import coco_items  # noqa: E402
from s8_other_models import MODELS  # noqa: E402

from wspd import config  # noqa: E402
from wspd.model import best_device  # noqa: E402
from wspd.splits import dev_items  # noqa: E402

torch.set_grad_enabled(False)
BASE, VAE = "stabilityai/stable-diffusion-xl-base-1.0", "madebyollin/sdxl-vae-fp16-fix"
STEPS = 4
ROOT = config.RUNS / "study" / "quality_sdxl"
# host -> (sampler, extra arguments of the pipeline call), following each release's instructions
HOSTS = {
    "lcm": ("lcm", {}),
    "tcd": ("tcd", {"eta": 0.3}),
    "hyper4": ("ddim", {}),
    "dmd2": ("lcm", {"timesteps": [999, 749, 499, 249]}),
}
SAMPLERS = {
    "lcm": lambda cfg: LCMScheduler.from_config(cfg),
    "tcd": lambda cfg: TCDScheduler.from_config(cfg),
    "ddim": lambda cfg: DDIMScheduler.from_config(cfg, timestep_spacing="trailing"),
}


def load_factors(pipe, host: str):
    """The UNet part of an SDXL adapter, per dotted module name, through diffusers' own key conversion."""
    cache = config.RUNS / "study" / f"factors_sdxlfn_{host}.pt"
    if cache.exists():
        return torch.load(cache)
    a = MODELS["sdxl"][host]
    state, alphas = pipe.lora_state_dict(hf_hub_download(a.repo, a.file, revision=a.rev), unet_config=pipe.unet.config)
    alphas = alphas or {}
    out = {}
    for k in sorted(state):
        # Two key layouts come out of the conversion: "<module>.lora.down.weight" and, for attention
        # layers, "<attention>.processor.to_k_lora.down.weight" (to_out stands for to_out.0).
        m = re.fullmatch(r"unet\.(.+)\.lora\.down\.weight", k)
        p = re.fullmatch(r"unet\.(.+)\.processor\.(to_[a-z]+)_lora\.down\.weight", k)
        if m:
            name, alpha = m.group(1), alphas.get(f"unet.{m.group(1)}.alpha")
        elif p:
            name, alpha = f"{p.group(1)}.{p.group(2)}" + (".0" if p.group(2) == "to_out" else ""), alphas.get(f"{k}.alpha")
        else:
            assert ".up.weight" in k, f"unrecognised key {k}"
            continue
        down, up = state[k].float(), state[k.replace(".down.weight", ".up.weight")].float()
        scale = float(alpha) / down.shape[0] if alpha is not None else 1.0
        assert tuple(pipe.unet.get_submodule(name).weight.shape[:2]) == (up.shape[0], down.shape[1]), name
        out[name] = factorize(up, down, scale)
    assert out, f"no UNet LoRA layers found for {host}"
    torch.save(out, cache)
    return out


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    block = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    device = best_device()
    vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(BASE, vae=vae, torch_dtype=torch.float16, variant="fp16").to(device)
    pipe.set_progress_bar_config(disable=True)
    sched_cfg = pipe.scheduler.config
    unet = pipe.unet
    weights = unet.state_dict()
    pristine: dict[str, torch.Tensor] = {}
    state = {"step": 0}
    unet.register_forward_hook(lambda m, a, out: state.update(step=min(state["step"] + 1, STEPS - 1)))

    def install(parts) -> None:
        for name, w in pristine.items():
            weights[f"{name}.weight"].copy_(w)
        for name, (u, s, vh) in (parts or {}).items():
            target = weights[f"{name}.weight"]
            if name not in pristine:
                pristine[name] = target.detach().cpu().clone()
            target.add_(((u.to(device) * s.to(device)) @ vh.to(device)).view_as(target).to(target.dtype))
        torch.mps.empty_cache() if str(device) == "mps" else None

    def sample(prompt: str, seed: int, sampler: str, extra: dict):
        pipe.scheduler = SAMPLERS[sampler](sched_cfg)
        state["step"] = 0
        return pipe(prompt=prompt, num_inference_steps=STEPS, guidance_scale=0.0, height=1024, width=1024, generator=torch.Generator("cpu").manual_seed(seed), **extra).images[0]

    factors = {h: load_factors(pipe, h) for h in HOSTS}
    for h, f in factors.items():
        top = sum(float(s[0] ** 2) for _, s, _ in f.values()) / sum(float((s**2).sum()) for _, s, _ in f.values())
        print(f"{h:8s} {len(f)} layers | top-1 energy {top:.3f}", flush=True)

    def fit_constants(host: str) -> dict:
        """Mean of v.x per layer over positions, steps and the development prompts, with the rank-1 adapter running."""
        cache = config.RUNS / "study" / f"constants_sdxl_{host}.pt"
        if cache.exists():
            return torch.load(cache)
        f = factors[host]
        total, count = defaultdict(float), defaultdict(int)

        def record(name, v):
            def hook(module, args, output):
                c = coefficient(module, args[0].float(), v)
                total[name] += float(c.sum())
                count[name] += c.numel()
            return hook

        install(rank1(f))
        handles = [unet.get_submodule(name).register_forward_hook(record(name, f[name][2][0].to(device))) for name in f]
        for it in dev_items(16):
            sample(it.prompt, it.seed, HOSTS[host][0], HOSTS[host][1])
        for h in handles:
            h.remove()
        install(None)
        out = {name: total[name] / count[name] for name in f}
        torch.save(out, cache)
        print(f"constants {host}", flush=True)
        return out

    consts = {h: fit_constants(h) for h in HOSTS}

    def shift_hooks(host: str):
        handles = []
        for name, (u, s, _) in factors[host].items():
            module = unet.get_submodule(name)
            vec = (shaped(module, u[:, 0].to(device)) * (float(s[0]) * consts[host][name])).to(torch.float16)
            handles.append(module.register_forward_hook(lambda m, a, out, vec=vec: out + vec))
        return handles

    arms = {f"none_{s}": (s, {}, None, None) for s in sorted({v[0] for v in HOSTS.values()})}
    for h, (sampler, extra) in HOSTS.items():
        arms[f"{h}_full"] = (sampler, extra, factors[h], None)
        arms[f"{h}_rank1"] = (sampler, extra, rank1(factors[h]), None)
        arms[f"{h}_const"] = (sampler, extra, None, lambda h=h: shift_hooks(h))

    items = coco_items(n)
    for start in range(0, n, block):
        for name, (sampler, extra, parts, hooks) in arms.items():
            folder = ROOT / name
            folder.mkdir(parents=True, exist_ok=True)
            todo = [(i, it) for i, it in enumerate(items[start : start + block], start) if not (folder / f"{i:05d}.png").exists()]
            if not todo:
                continue
            t0 = time.time()
            install(parts)
            handles = hooks() if hooks else []
            try:
                for i, it in todo:
                    sample(it["caption"], it["image_id"], sampler, extra).resize((512, 512), Image.LANCZOS).save(folder / f"{i:05d}.png")
            finally:
                for h in handles:
                    h.remove()
                install(None)
            print(f"block {start:5d} | {name:14s} | {len(todo)} images | {time.time() - t0:5.0f} s", flush=True)


if __name__ == "__main__":
    main()
