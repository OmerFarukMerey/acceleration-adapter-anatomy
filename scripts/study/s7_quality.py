"""Study 7: image quality at scale, on MS-COCO 2014 validation captions.

Generates one 512x512 image per caption for every arm and saves it under runs/study/quality/.
Arms, all on Stable Diffusion 1.5:
  ref_base25                     base model, 25 steps, guidance 7.5 (the many-step reference)
  base4_guided                   base model, 4 steps, guidance 7.5 (what you get without an adapter)
  none_<sampler>                 base model, 4 steps, no guidance, with each adapter's sampler
  <host>_full                    the adapter as released
  <host>_rank1                   only its top direction per layer
  <host>_const_step              one fixed vector per layer and step added to the layer's output
  <host>_const                   one fixed vector per layer, the same at every step
The fixed vectors are estimated on the 16 development prompts, never on COCO captions.
Work is done in blocks of captions across all arms, so a stopped run leaves balanced arms.

    python scripts/study/s7_quality.py [n_captions] [block]
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from adapters import ADAPTERS, SAMPLERS, bench, load_factors, rank1  # noqa: E402
from s3_mechanism import coefficient, shaped  # noqa: E402

from wspd import config  # noqa: E402

HOSTS = ["lcm", "tcd", "pcm", "hyper4", "flash"]
STEPS = 4
ROOT = config.RUNS / "study" / "quality"
CAPTIONS = config.RUNS / "study" / "coco" / "annotations" / "captions_val2014.json"


def coco_items(n: int) -> list[dict]:
    """The first n images of a fixed random order of COCO val2014, each with its first caption."""
    data = json.loads(CAPTIONS.read_text())
    first: dict[int, tuple[int, str]] = {}
    for a in data["annotations"]:
        if a["image_id"] not in first or a["id"] < first[a["image_id"]][0]:
            first[a["image_id"]] = (a["id"], a["caption"].strip())
    files = {im["id"]: im["file_name"] for im in data["images"]}
    ids = sorted(first)
    order = np.random.default_rng(0).permutation(len(ids))[:n]
    return [{"image_id": ids[i], "caption": first[ids[i]][1], "file_name": files[ids[i]]} for i in order]


def fit_constants(b, host: str) -> dict:
    """Mean of v.x per layer and step, on the development prompts, with the rank-1 adapter running."""
    cache = config.RUNS / "study" / f"constants_{host}.pt"
    if cache.exists():
        return torch.load(cache)
    f = load_factors(host)
    mods = {n: b.unet.get_submodule(n) for n in f}
    state = {"step": 0}
    total, count = defaultdict(float), defaultdict(int)

    def record(n, v):
        def hook(module, args, output):
            c = coefficient(module, args[0], v)
            total[n, state["step"]] += float(c.sum())
            count[n, state["step"]] += c.numel()
        return hook

    b.install(rank1(f))
    handles = [m.register_forward_hook(record(n, f[n][2][0].to(b.device))) for n, m in mods.items()]
    for _ in b.trajectory(ADAPTERS[host].sampler, STEPS, before=lambda i, step: state.update(step=step)):
        pass
    for h in handles:
        h.remove()
    b.install(None)
    out = {n: [total[n, s] / count[n, s] for s in range(STEPS)] for n in f}
    torch.save(out, cache)
    return out


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    block = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    items = coco_items(n)
    (config.RESULTS / "study" / "s7_captions.json").write_text(json.dumps(items, indent=1))
    b = bench()
    device = b.device
    factors = {h: load_factors(h) for h in HOSTS}
    consts = {h: fit_constants(b, h) for h in HOSTS}
    state = {"step": 0}

    def shift_hooks(host: str, per_step: bool):
        f, c = factors[host], consts[host]
        handles = []
        for name in f:
            module = b.unet.get_submodule(name)
            u, s = shaped(module, f[name][0][:, 0].to(device)), float(f[name][1][0])
            values = c[name] if per_step else [float(np.mean(c[name]))] * STEPS
            handles.append(module.register_forward_hook(lambda m, a, out, u=u, s=s, values=values: out + u * (s * values[state["step"]])))
        return handles

    # name -> (sampler, steps, guidance, weight change, hook factory)
    arms = {"ref_base25": ("ddim", 25, 7.5, None, None), "base4_guided": ("ddim", STEPS, 7.5, None, None)}
    for sampler in sorted({ADAPTERS[h].sampler for h in HOSTS}):
        arms[f"none_{sampler}"] = (sampler, STEPS, 1.0, None, None)
    for h in HOSTS:
        sampler = ADAPTERS[h].sampler
        arms[f"{h}_full"] = (sampler, STEPS, 1.0, factors[h], None)
        arms[f"{h}_rank1"] = (sampler, STEPS, 1.0, rank1(factors[h]), None)
        arms[f"{h}_const_step"] = (sampler, STEPS, 1.0, None, lambda h=h: shift_hooks(h, True))
        arms[f"{h}_const"] = (sampler, STEPS, 1.0, None, lambda h=h: shift_hooks(h, False))

    def sample(item: dict, sampler: str, steps: int, guidance: float) -> np.ndarray:
        cond = b.encode(item["caption"])
        sched = SAMPLERS[sampler](b.sched_cfg)
        sched.set_timesteps(steps, device=device)
        seed = item["image_id"]
        z = torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(seed)).to(device) * sched.init_noise_sigma
        noise = torch.Generator("cpu").manual_seed(seed + 1)
        for step, t in enumerate(sched.timesteps):
            state["step"] = min(step, STEPS - 1)
            zin = sched.scale_model_input(z, t)
            eps = b.unet(zin, t, encoder_hidden_states=cond).sample
            if guidance != 1.0:
                eps_un = b.unet(zin, t, encoder_hidden_states=b.empty).sample
                eps = eps_un + guidance * (eps - eps_un)
            extra = {"generator": noise} if sampler != "ddim" else {}
            z = sched.step(eps, t, z, **extra, return_dict=False)[0]
        x = b.vae.decode(z / b.vae.config.scaling_factor).sample
        return ((x.clamp(-1, 1) + 1) * 127.5).round().permute(0, 2, 3, 1).byte().cpu().numpy()[0]

    for start in range(0, n, block):
        for name, (sampler, steps, guidance, parts, hooks) in arms.items():
            folder = ROOT / name
            folder.mkdir(parents=True, exist_ok=True)
            todo = [(i, it) for i, it in enumerate(items[start : start + block], start) if not (folder / f"{i:05d}.png").exists()]
            if not todo:
                continue
            t0 = time.time()
            b.install(parts)
            handles = hooks() if hooks else []
            try:
                for i, it in todo:
                    Image.fromarray(sample(it, sampler, steps, guidance)).save(folder / f"{i:05d}.png")
            finally:
                for h in handles:
                    h.remove()
            print(f"block {start:5d} | {name:18s} | {len(todo)} images | {time.time() - t0:6.0f} s", flush=True)
    b.install(None)


if __name__ == "__main__":
    main()
