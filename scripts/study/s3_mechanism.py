"""Study 3: what does the shared direction do?

Two tests per adapter, at 4 steps with its own sampler.

A. Guidance. On the adapter's own sampling path, the change it makes to the network output
   (adapter minus base, same input and prompt) is compared with the base model's guidance
   direction (prompted minus unprompted output). If the adapter were mostly built-in
   guidance the two would line up.

B. Bias. A rank-1 change adds `s * u * (v . x)` to a layer's output. If `v . x` hardly varies,
   the adapter is close to a constant added to each layer. The coefficient is replaced by
   its average over image positions, then by one constant per layer and step, then by one
   constant per layer. Constants are estimated on half of the prompts; recovery is reported
   on the other half too.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pilots"))
from adapters import ADAPTERS, bench, load_factors  # noqa: E402
from c_lora_anatomy import line, score, sheet  # noqa: E402

from wspd import config  # noqa: E402
from wspd.metrics import Evaluator  # noqa: E402
from wspd.store import device_lock  # noqa: E402

HOSTS = sys.argv[1:] or ["lcm", "tcd", "pcm", "hyper4", "flash"]
STEPS = 4


def cos(a, b) -> float:
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def guidance_test(b, full, sampler) -> dict:
    """Output-level comparison of the adapter's effect with the base model's guidance direction."""
    records = []
    b.install(full)
    for _ in b.trajectory(sampler, STEPS, visit=lambda i, step, t, z, eps, _: records.append((i, step, t, z.clone(), eps.clone()))):
        pass
    b.install(None)
    rows = defaultdict(list)
    effects, guides = {}, {}
    for i, step, t, z, eps_adapter in records:
        eps_c = b.unet(z, t, encoder_hidden_states=b.conds[i]).sample
        eps_u = b.unet(z, t, encoder_hidden_states=b.empty).sample
        effect, guide = (eps_adapter - eps_c).flatten().cpu().double(), (eps_c - eps_u).flatten().cpu().double()
        effects[i, step], guides[i, step] = effect, guide
        basis = torch.stack([guide, eps_c.flatten().cpu().double()], 1)  # guidance, and rescaling of the prediction
        coef = torch.linalg.lstsq(basis, effect[:, None]).solution[:, 0]
        rows[step].append({
            "cos_guidance": cos(effect, guide),
            "extra_guidance_scale": float(effect @ guide / (guide @ guide)),
            "relative_size": float(effect.norm()) / float(eps_c.norm()),
            "explained_by_guidance_and_rescaling": float(1 - (effect - basis @ coef).pow(2).sum() / effect.pow(2).sum()),
            "rescaling_coefficient": float(coef[1]),
        })
    n = len(b.prompts)
    for (i, step), effect in effects.items():
        rows[step][i]["cos_guidance_other_prompt"] = cos(effect, guides[(i + 1) % n, step])
    return {f"step_{s}": {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} for s, rs in sorted(rows.items())}


def coefficient(module, x, v):
    """v . x for every output position of the layer."""
    if isinstance(module, torch.nn.Conv2d):
        return F.conv2d(x, v.view(1, *module.weight.shape[1:]), stride=module.stride, padding=module.padding)
    return F.linear(x, v[None])


def shaped(module, u):
    return u.view(1, -1, 1, 1) if isinstance(module, torch.nn.Conv2d) else u


def bias_test(b, f, sampler, evaluator) -> dict:
    device = b.device
    mods = {n: b.unet.get_submodule(n) for n in f}
    top = {n: (u[:, 0].to(device), float(s[0]), vh[0].to(device)) for n, (u, s, vh) in f.items()}
    rank1 = {n: (u[:, :1], s[:1], vh[:1]) for n, (u, s, vh) in f.items()}
    state = {"step": 0, "prompt": 0}
    before = lambda i, step: state.update(step=step, prompt=i)  # noqa: E731
    fit = lambda: state["prompt"] % 2 == 0  # noqa: E731  constants come from the even-numbered prompts

    def run(parts, hook=None):
        b.install(parts)
        handles = [m.register_forward_hook(hook(n)) for n, m in mods.items()] if hook else []
        try:
            # generate() has no callback, so decode from trajectory() here.
            return decode(b, list(b.trajectory(sampler, STEPS, before=before)))
        finally:
            for h in handles:
                h.remove()

    # Record the coefficient's first and second moments on the working rank-1 model.
    total, square, count = defaultdict(float), defaultdict(float), defaultdict(int)

    def record(n):
        def hook(module, args, output):
            if fit():
                c = coefficient(module, args[0], top[n][2])
                key = (n, state["step"])
                total[key] += float(c.sum())
                square[key] += float(c.pow(2).sum())
                count[key] += c.numel()
        return hook

    images = {"none": run(None), "full": run(f), "rank_1": run(rank1, record)}
    mean = {k: total[k] / count[k] for k in total}
    per_layer = {n: float(np.mean([mean[n, s] for s in range(STEPS)])) for n in f}

    def add(value):
        def make(n):
            u, s, v = top[n]
            def hook(module, args, output):
                return output + shaped(module, u) * (s * value(n, module, args[0], v))
            return hook
        return make

    def positions_averaged(n, module, x, v):
        c = coefficient(module, x, v)
        return c.mean(dim=tuple(range(1, c.dim())), keepdim=True)

    images["positions_averaged"] = run(None, add(positions_averaged))
    images["constant_per_step"] = run(None, add(lambda n, module, x, v: mean[n, state["step"]]))
    images["constant"] = run(None, add(lambda n, module, x, v: per_layer[n]))
    b.install(None)

    held_out = np.arange(len(b.prompts)) % 2 == 1
    result = score(evaluator, images, b.prompts)
    held = score(evaluator, {k: v[held_out] for k, v in images.items()}, [p for p, h in zip(b.prompts, held_out) if h])
    for name, r in result.items():
        r["recovery_held_out"], r["clip_held_out"] = held[name]["recovery"], held[name]["clip"]
    # How constant the coefficient is: share of its mean square that the per-step mean explains.
    weight = {n: top[n][1] ** 2 for n in f}
    share = sum(weight[n] * np.mean([mean[n, s] ** 2 / (square[n, s] / count[n, s]) for s in range(STEPS)]) for n in f) / sum(weight.values())
    return {"arms": result, "coefficient_constancy": float(share), "images": images}


def decode(b, latents) -> np.ndarray:
    out = []
    for z in latents:
        x = decode.vae.decode(z / decode.vae.config.scaling_factor).sample
        out.append(((x.clamp(-1, 1) + 1) * 127.5).round().permute(0, 2, 3, 1).byte().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    from diffusers import AutoencoderKL
    from adapters import BASE, BASE_REV

    b = bench()
    decode.vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae", revision=BASE_REV, torch_dtype=torch.float32).to(b.device).eval()
    evaluator = Evaluator(b.device)
    out = config.RESULTS / "study"
    path = out / "s3_mechanism.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    show = ["none", "constant", "constant_per_step", "positions_averaged", "rank_1", "full"]
    for host in HOSTS:
        f = load_factors(host)
        guidance = guidance_test(b, f, ADAPTERS[host].sampler)
        for step, r in guidance.items():
            print(f"{host:7s} {step} | cos with guidance {r['cos_guidance']:+.3f} (other prompt {r['cos_guidance_other_prompt']:+.3f}) | extra scale {r['extra_guidance_scale']:+.2f} | size {r['relative_size']:.2f} | explained {r['explained_by_guidance_and_rescaling']:.2f} | rescale {r['rescaling_coefficient']:+.2f}", flush=True)
        bias = bias_test(b, f, ADAPTERS[host].sampler, evaluator)
        for name, r in bias["arms"].items():
            print(f"{host:7s} {line(name, r)} | held-out recovery {r['recovery_held_out']:+.2f}", flush=True)
        print(f"{host:7s} coefficient constancy {bias['coefficient_constancy']:.3f}", flush=True)
        sheet(bias.pop("images"), show, out / f"s3_mechanism_{host}.png")
        results[host] = {"guidance": guidance, "bias": bias, "sheet_columns": show}
        path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    with device_lock():
        main()
