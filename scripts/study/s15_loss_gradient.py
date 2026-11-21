"""Study 15: does a real method's own loss point the same way at the base weights?

Study 14 used a plain teacher-matching loss. Here the loss is latent consistency distillation
as in the diffusers LCM-LoRA training script (train_lcm_distill_lora_sd_wds.py) with its
default settings: a guided teacher takes one DDIM solver step from timestep t+k to t (50
solver steps, guidance scale drawn from 5 to 15), and the student is asked to predict the
same clean latent from both ends (squared error, LCM's boundary scalings). At the base
weights the LoRA is zero, so student, teacher and target network are all the base model and
no trained adapter is involved.

The gradient is averaged over COCO photos with their captions (items 500 onward of the fixed
order, used for no score), with one random timestep, noise and guidance scale each. Its top
direction per layer is compared with every adapter's, with the direction they share and with
study 14's direction.

The loss is also evaluated with each adapter installed, and along the line from the base
model to LCM-LoRA, to see whether the released adapters sit at a lower value of it.

    python scripts/study/s15_loss_gradient.py [n_images] [n_images_for_the_check]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler

sys.path.insert(0, str(Path(__file__).parent))
from adapters import ACCEL, STYLE, load_factors  # noqa: E402
from s1_weights import compare  # noqa: E402
from s2_function import HOSTS, consensus  # noqa: E402
from s7_quality import coco_items  # noqa: E402
from s7_score import real_image  # noqa: E402
from s14_gradient import CACHE as PLAIN  # noqa: E402
from s14_gradient import FULL, model, top_factors, unit  # noqa: E402

from wspd import config  # noqa: E402

N_SOLVER, W_MIN, W_MAX, SCALING, SIGMA_DATA = 50, 5.0, 15.0, 10.0, 0.5
TOPK = 1000 // N_SOLVER
FIRST = 500  # the first 500 items are the evaluation captions of study 7
CACHE = config.RUNS / "study" / "factors_gradient_lcd.pt"


def boundary(t: int) -> tuple[float, float]:
    """LCM's skip and output scalings, which make the prediction the input itself at t = 0."""
    s = SCALING * t
    return SIGMA_DATA**2 / (s**2 + SIGMA_DATA**2), s / (s**2 + SIGMA_DATA**2) ** 0.5


def clean(eps, z, a):
    """The clean latent implied by a noise prediction."""
    return (z - (1 - a).sqrt() * eps) / a.sqrt()


class Data:
    """Photo latents, captions and the random draws of the loss, fixed per image."""

    def __init__(self, n: int):
        self.b = b = model()
        self.items = coco_items(FIRST + n)[FIRST:]
        self.alphas = DDIMScheduler.from_config(b.sched_cfg).alphas_cumprod.to(b.device)
        self.solver_t = torch.arange(1, N_SOLVER + 1) * TOPK - 1
        cache = config.RUNS / "study" / f"coco_latents_{FIRST}_{n}.pt"
        if cache.exists():
            self.latents = torch.load(cache)
        else:
            self.latents = []
            for it in self.items:
                x = torch.from_numpy(np.asarray(real_image(it))).float().div(127.5).sub(1).permute(2, 0, 1)[None]
                self.latents.append((b.vae.encode(x.to(b.device)).latent_dist.mode() * b.vae.config.scaling_factor).cpu())
            torch.save(self.latents, cache)
        self.draws = []
        for it in self.items:
            gen = torch.Generator().manual_seed(it["image_id"])
            self.draws.append((int(torch.randint(0, N_SOLVER, (1,), generator=gen)), torch.randn((1, 4, 64, 64), generator=gen), float(W_MIN + (W_MAX - W_MIN) * torch.rand((), generator=gen))))

    def start(self, i: int):
        """The noised latent at t+k, with both timesteps and their signal levels."""
        index, noise, _ = self.draws[i]
        t_start = int(self.solver_t[index])
        t = max(t_start - TOPK, 0)
        a_start, a = self.alphas[t_start], self.alphas[t]
        z = a_start.sqrt() * self.latents[i].to(self.b.device) + (1 - a_start).sqrt() * noise.to(self.b.device)
        return z, t_start, t, a_start, a

    def predict(self, z, t: int, a, cond, eps=None):
        """The consistency function: boundary scalings around the clean latent the network implies."""
        if eps is None:
            eps = self.b.unet(z, t, encoder_hidden_states=cond).sample
        c_skip, c_out = boundary(t)
        return c_skip * z + c_out * clean(eps, z, a)


def gradient(n: int) -> dict:
    """Factors of minus the loss gradient per layer, from all images and from each half, with the solver's outputs."""
    if CACHE.exists():
        return torch.load(CACHE)
    d = Data(n)
    b, device = d.b, d.b.device
    weights = {name: b.unet.get_submodule(name).weight for name in sorted(load_factors("lcm"))}
    b.unet.requires_grad_(False)
    for w in weights.values():
        w.requires_grad_(True)
    total = {name: torch.zeros(w.shape) for name, w in weights.items()}
    out = {"n": n, "halves": [], "loss": [0.0] * n, "t_start": [0] * n, "z_prev": [None] * n}
    t0 = time.time()
    for half in (0, 1):
        for i in range(half, n, 2):
            z, t_start, t, a_start, a = d.start(i)
            cond, guidance = b.encode(d.items[i]["caption"]), d.draws[i][2]
            with torch.enable_grad():
                eps = b.unet(z, t_start, encoder_hidden_states=cond).sample
            # The teacher is the same network here, so its prompted output is the student's.
            eps_c, eps_u = eps.detach(), b.unet(z, t_start, encoder_hidden_states=b.empty).sample
            x0 = clean(eps_c, z, a_start) + guidance * (clean(eps_c, z, a_start) - clean(eps_u, z, a_start))
            z_prev = a.sqrt() * x0 + (1 - a).sqrt() * (eps_c + guidance * (eps_c - eps_u))
            target = d.predict(z_prev, t, a, cond)
            with torch.enable_grad():
                loss = F.mse_loss(d.predict(z, t_start, a_start, cond, eps), target)
                loss.backward()  # gradients add up over the images
            out["loss"][i], out["t_start"][i], out["z_prev"][i] = float(loss), t_start, z_prev.cpu()
        grads = {name: w.grad.detach().cpu() for name, w in weights.items()}
        for name, w in weights.items():
            w.grad = None
            total[name] += grads[name]
        out["halves"].append({name: top_factors(-g)[0] for name, g in grads.items()})
        print(f"half {half} | mean loss {np.mean(out['loss'][half::2]):.5f} | {time.time() - t0:5.0f} s", flush=True)
    b.unet.requires_grad_(False)
    both = {name: top_factors(-g) for name, g in total.items()}
    out["sum"], out["share_sum"] = {k: v[0] for k, v in both.items()}, {k: v[1] for k, v in both.items()}
    torch.save(out, CACHE)
    return out


def loss_with(d: Data, g: dict, parts, count: int) -> np.ndarray:
    """The loss per image with a weight change installed in student and target; the teacher stays the base model."""
    b = d.b
    b.install(parts)
    losses = []
    for i in range(count):
        z, t_start, t, a_start, a = d.start(i)
        cond = b.encode(d.items[i]["caption"])
        losses.append(float(F.mse_loss(d.predict(z, t_start, a_start, cond), d.predict(g["z_prev"][i].to(b.device), t, a, cond))))
    b.install(None)
    return np.array(losses)


def loss_along(d: Data, g: dict, parts, steps: tuple, count: int) -> dict:
    """The loss with a fraction of an adapter installed: in student and target, as in training, and with the
    target left at the base model, which is the case the gradient describes."""
    b = d.b
    conds = [b.encode(d.items[i]["caption"]) for i in range(count)]
    starts = [d.start(i) for i in range(count)]
    base_target = [d.predict(g["z_prev"][i].to(b.device), starts[i][2], starts[i][4], conds[i]) for i in range(count)]
    out = {}
    for fraction in steps:
        b.install({name: (u, s * fraction, vh) for name, (u, s, vh) in parts.items()})
        both, fixed = [], []
        for i, (z, t_start, t, a_start, a) in enumerate(starts):
            pred = d.predict(z, t_start, a_start, conds[i])
            both.append(float(F.mse_loss(pred, d.predict(g["z_prev"][i].to(b.device), t, a, conds[i]))))
            fixed.append(float(F.mse_loss(pred, base_target[i])))
        out[str(fraction)] = {"adapter_in_student_and_target": float(np.mean(both)), "target_at_base": float(np.mean(fixed))}
        print(f"fraction {fraction:5.2f} of the adapter | loss {np.mean(both):.5f} | with the target at the base model {np.mean(fixed):.5f}", flush=True)
    b.install(None)
    return out


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    g = gradient(n)
    factors = {k: load_factors(k) for k in [*ACCEL, *STYLE]}
    targets = {"shared": consensus([factors[h] for h in HOSTS], factors["lcm"]), **factors}
    for key in FULL:
        path = config.RUNS / "study" / f"factors_{key}.pt"
        if path.exists():
            targets[key] = {name: v[:3] for name, v in torch.load(path).items()}
    if PLAIN.exists():
        targets["plain_loss_gradient"] = unit(torch.load(PLAIN)["sum"])
    keep = ("layers", "out_cos_weighted", "out_cos_median", "in_cos_weighted", "random_out_cos", "mismatched_out_cos_weighted", "same_sign_share", "delta_cos_global", "by_block", "by_kind")
    path = config.RESULTS / "study" / "s15_loss_gradient.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    results.update({
        "images": n, "solver_steps": N_SOLVER, "guidance_range": [W_MIN, W_MAX],
        "loss_at_base": float(np.mean(g["loss"])),
        "gradient_top1_energy_share_median": float(np.median(list(g["share_sum"].values()))),
        "pairs": {},
    })
    for name, f in targets.items():
        results["pairs"][name] = r = {k: v for k, v in compare(unit(g["sum"]), f).items() if k in keep}
        print(f"consistency-loss gradient vs {name:20s} out {r['out_cos_weighted']:.3f} (median {r['out_cos_median']:.3f}, random {r['random_out_cos']:.3f}, mismatched {r['mismatched_out_cos_weighted']:.3f}) | in {r['in_cos_weighted']:.3f} | same sign {r['same_sign_share']:.2f} | delta {r['delta_cos_global']:+.3f}", flush=True)
    halves = [unit(h) for h in g["halves"]]
    r = compare(halves[0], halves[1])
    results["split_half"] = {"even_vs_odd_images": {k: r[k] for k in ("out_cos_weighted", "out_cos_median", "in_cos_weighted")}, "each_half_vs_shared": [compare(h, targets["shared"])["out_cos_weighted"] for h in halves]}
    print(f"gradient from even against odd images: out {r['out_cos_weighted']:.3f} (median {r['out_cos_median']:.3f}) | each half vs shared {[round(x, 3) for x in results['split_half']['each_half_vs_shared']]}", flush=True)
    path.write_text(json.dumps(results, indent=1))

    # Does each adapter lower this loss? Same images, timesteps, noise and solver outputs for all.
    results.setdefault("loss_with_adapter", {})
    todo = [k for k in ["none", *HOSTS, *STYLE] if k not in results["loss_with_adapter"]]
    if todo:
        d = Data(n)
        base = None
        for key in ["none", *todo] if "none" not in todo else todo:
            losses = loss_with(d, g, None if key == "none" else factors[key], count)
            if key == "none":
                base = losses
            results["loss_with_adapter"][key] = {"images": count, "mean": float(losses.mean()), "relative_to_base": float(losses.mean() / base.mean()), "share_of_images_lower_than_base": float((losses < base).mean())}
            print(f"loss with {key:14s} {losses.mean():.5f} | {losses.mean() / base.mean():.2f} of the base model's | lower on {(losses < base).mean():.0%} of images", flush=True)
            path.write_text(json.dumps(results, indent=1))

    # The full adapters raise the loss. Is that so from the first small step, or only far along?
    if "loss_along_lcm" not in results:
        results["loss_along_lcm"] = loss_along(Data(n), g, factors["lcm"], (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0), count // 2)
        path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
