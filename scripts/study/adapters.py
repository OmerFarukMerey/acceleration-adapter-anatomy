"""Shared code for the cross-method study: the adapter registry, factorization and a sampling bench.

Every adapter is a LoRA for Stable Diffusion 1.5. Each is loaded as, per UNet layer, the
singular value decomposition of its weight change: (u, s, vh) with delta = (u * s) @ vh.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, LCMScheduler, TCDScheduler, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer

from wspd import config
from wspd.model import best_device
from wspd.splits import dev_items

torch.set_grad_enabled(False)
BASE, BASE_REV = "stable-diffusion-v1-5/stable-diffusion-v1-5", "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
BLOCKS = ("down_blocks", "mid_block", "up_blocks")
KINDS = ("time", "conv", "self_attn", "cross_attn", "ff", "attn_proj")
Factors = dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class Adapter:
    repo: str
    file: str
    rev: str
    family: str  # the training objective, or "style" for the non-acceleration controls
    sampler: str = "lcm"
    steps: int = 4


_HYPER = ("ByteDance/Hyper-SD", "bc08d970a87c74c71209491d64e3525845698863")
_PCM = ("wangfuyun/PCM_Weights", "39560fead4ce00f94db3cb8e93dd8fba90ec0be6")
ADAPTERS = {
    "lcm": Adapter("latent-consistency/lcm-lora-sdv1-5", "pytorch_lora_weights.safetensors", "cf2fced511dbe7e26c8d1d397e728fbab875db4b", "consistency", "lcm"),
    "tcd": Adapter("h1t/TCD-SD15-LoRA", "pytorch_lora_weights.safetensors", "449a99f6bbded2bdbbc66386b2a4fbf8db8d3807", "trajectory consistency", "tcd"),
    "pcm": Adapter(_PCM[0], "sd15/pcm_sd15_smallcfg_4step_converted.safetensors", _PCM[1], "phased consistency + adversarial", "ddim"),
    "pcm_lcmlike": Adapter(_PCM[0], "sd15/pcm_sd15_lcmlike_lora_converted.safetensors", _PCM[1], "phased consistency + adversarial", "lcm"),
    "hyper1": Adapter(_HYPER[0], "Hyper-SD15-1step-lora.safetensors", _HYPER[1], "segmented consistency + adversarial + preference", "tcd", 1),
    "hyper2": Adapter(_HYPER[0], "Hyper-SD15-2steps-lora.safetensors", _HYPER[1], "segmented consistency + adversarial + preference", "ddim", 2),
    "hyper4": Adapter(_HYPER[0], "Hyper-SD15-4steps-lora.safetensors", _HYPER[1], "segmented consistency + adversarial + preference", "ddim", 4),
    "hyper8": Adapter(_HYPER[0], "Hyper-SD15-8steps-lora.safetensors", _HYPER[1], "segmented consistency + adversarial + preference", "ddim", 8),
    "flash": Adapter("jasperai/flash-sd", "pytorch_lora_weights.safetensors", "4c52101c931beb11196d183b89ca525acb80e8fe", "distribution matching + adversarial", "lcm"),
    "style_monet_a": Adapter("neonforestmist/sd15-monet-lora", "monet_style_lora.safetensors", "b3078d418319cdd2c8896f36687f7bc22548bfbe", "style"),
    "style_monet_b": Adapter("J-YOON/lora-monet-sd1.5", "pytorch_lora_weights.safetensors", "6b5fb2df5a63fc01ec99c44c90b3d2dd54a4abb4", "style"),
    "style_pixel": Adapter("artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5", "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors", "eb7c3bfe0c48e7fadd635a0cfb25c29b428c4e63", "style"),
}
ACCEL = [k for k, a in ADAPTERS.items() if a.family != "style"]
STYLE = [k for k, a in ADAPTERS.items() if a.family == "style"]


def kind(name: str) -> str:
    if name.endswith("time_emb_proj"):
        return "time"
    if ".attn1." in name:
        return "self_attn"
    if ".attn2." in name:
        return "cross_attn"
    if ".ff." in name:
        return "ff"
    if name.endswith(("proj_in", "proj_out")):
        return "attn_proj"
    return "conv"


def block(name: str) -> str:
    return name.split(".")[0]


def factorize(up: torch.Tensor, down: torch.Tensor, scale: float):
    """SVD of scale * up @ down through two thin QRs, so truncation is exact and cheap."""
    qu, ru = torch.linalg.qr(up.flatten(1).double())
    qd, rd = torch.linalg.qr(down.flatten(1).double().T)
    u, s, vh = torch.linalg.svd(ru @ rd.T)
    return (qu @ u).float(), (scale * s).float(), (qd @ vh.T).T.float()


def unet_layer_names() -> dict[str, str]:
    """Maps the flattened layer names used in LoRA files to dotted UNet module names."""
    cfg = UNet2DConditionModel.load_config(BASE, subfolder="unet", revision=BASE_REV)
    unet = UNet2DConditionModel.from_config(cfg)
    return {n.replace(".", "_"): n for n, m in unet.named_modules() if hasattr(m, "weight") and m.weight is not None and m.weight.dim() >= 2}


def load_factors(key: str, names: dict[str, str] | None = None) -> Factors:
    """The UNet part of an adapter, factorized per layer. Cached under runs/."""
    cache = config.RUNS / "study" / f"factors_{key}.pt"
    if cache.exists():
        return torch.load(cache)
    a = ADAPTERS[key]
    raw = load_file(hf_hub_download(a.repo, a.file, revision=a.rev))
    names = names or unet_layer_names()
    out: Factors = {}
    for k in sorted(raw):
        m = re.fullmatch(r"lora_unet_(.+)\.lora_down\.weight", k) or re.fullmatch(r"unet\.(.+)\.lora\.down\.weight", k)
        if not m:
            continue
        stem = k[: -len("lora_down.weight")] if k.startswith("lora_unet_") else k[: -len("lora.down.weight")]
        up = raw[stem + ("lora_up.weight" if k.startswith("lora_unet_") else "lora.up.weight")]
        down = raw[k]
        alpha = raw.get(stem + "alpha")
        scale = float(alpha) / down.shape[0] if alpha is not None else 1.0
        out[names[m.group(1).replace(".", "_")]] = factorize(up, down, scale)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return out


def rank1(f: Factors) -> Factors:
    return {n: (u[:, :1], s[:1], vh[:1]) for n, (u, s, vh) in f.items()}


def dense(u, s, vh) -> torch.Tensor:
    return (u * s) @ vh


SAMPLERS = {
    "lcm": lambda cfg: LCMScheduler.from_config(cfg),
    "tcd": lambda cfg: TCDScheduler.from_config(cfg),
    "ddim": lambda cfg: DDIMScheduler.from_config(cfg, timestep_spacing="trailing"),
}


def bench(n_prompts: int = 16, model: str = BASE, revision: str = BASE_REV, variant: str | None = None) -> SimpleNamespace:
    """A model with closures to install a weight change and to sample. The base model unless another
    Stable Diffusion 1.5 checkpoint is named; its UNet, VAE and text encoder are then used, with the
    base model's tokenizer and noise schedule."""
    device = best_device()
    kw = dict(revision=revision, torch_dtype=torch.float32, use_safetensors=True, **({"variant": variant} if variant else {}))
    unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet", **kw).to(device).eval()
    vae = AutoencoderKL.from_pretrained(model, subfolder="vae", **kw).to(device).eval()
    text = CLIPTextModel.from_pretrained(model, subfolder="text_encoder", **kw).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer", revision=BASE_REV)
    sched_cfg = LCMScheduler.load_config(BASE, subfolder="scheduler", revision=BASE_REV)
    state = unet.state_dict()
    pristine: dict[str, torch.Tensor] = {}

    def install(parts: Factors | None) -> None:
        """Set the UNet to base + change. Each change is thin factors (u, s, vh); None restores the base.

        The dense change is formed one layer at a time on the device, so memory stays flat.
        """
        for name, weight in pristine.items():
            state[f"{name}.weight"].copy_(weight)
        for name, (u, s, vh) in (parts or {}).items():
            target = state[f"{name}.weight"]
            if name not in pristine:
                pristine[name] = target.detach().cpu().clone()
            target.add_(((u.to(device) * s.to(device)) @ vh.to(device)).view_as(target))
        if str(device) == "mps":
            torch.mps.empty_cache()

    items = dev_items(n_prompts)
    prompts = [it.prompt for it in items]

    def encode(p: str) -> torch.Tensor:
        ids = tok(p, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
        return text(ids.to(device))[0]

    conds, empty = [encode(p) for p in prompts], encode("")

    def trajectory(sampler: str, steps: int, guidance: float = 1.0, seed_offset: int = 0, visit=None, before=None):
        """Yields the final latent per prompt. before(i, step) runs ahead of each step's network call;
        visit(i, step, t, z, eps_cond, eps_uncond) sees its result."""
        for i, (it, cond) in enumerate(zip(items, conds)):
            sched = SAMPLERS[sampler](sched_cfg)
            sched.set_timesteps(steps, device=device)
            seed = it.seed + 1000 * seed_offset
            z = torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(seed)).to(device) * sched.init_noise_sigma
            noise = torch.Generator("cpu").manual_seed(seed + 1)
            for step, t in enumerate(sched.timesteps):
                if before:
                    before(i, step)
                zin = sched.scale_model_input(z, t)
                eps = unet(zin, t, encoder_hidden_states=cond).sample
                eps_un = None
                if guidance != 1.0:
                    eps_un = unet(zin, t, encoder_hidden_states=empty).sample
                if visit:
                    visit(i, step, t, zin, eps, eps_un)
                if guidance != 1.0:
                    eps = eps_un + guidance * (eps - eps_un)
                extra = {"generator": noise} if sampler != "ddim" else {}
                z = sched.step(eps, t, z, **extra, return_dict=False)[0]
            yield z

    def generate(sampler: str = "lcm", steps: int = 4, guidance: float = 1.0, seed_offset: int = 0) -> np.ndarray:
        out = []
        for z in trajectory(sampler, steps, guidance, seed_offset):
            x = vae.decode(z / vae.config.scaling_factor).sample
            out.append(((x.clamp(-1, 1) + 1) * 127.5).round().permute(0, 2, 3, 1).byte().cpu().numpy())
        return np.concatenate(out)

    return SimpleNamespace(device=device, unet=unet, vae=vae, encode=encode, sched_cfg=sched_cfg, install=install, generate=generate, trajectory=trajectory, prompts=prompts, items=items, conds=conds, empty=empty)
