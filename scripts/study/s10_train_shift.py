"""Study 10: can few-step sampling be had by training only a strength per layer? (a small pilot)

The base model is frozen. A 20-step guided DDIM run of the base model is the teacher. The
student takes 4 DDIM steps without guidance, and at each step is trained to land on the
teacher's state at the next student timestep (inputs come from the teacher's path).

What is trained, on the layers the LCM-LoRA touches:
  strength   one number per layer and step, multiplying a fixed direction added to the layer's
             output. The direction is the one the five methods' adapters agree on.
  random     the same, with a random fixed direction per layer (control for the direction)
  bias       a free vector per layer and step (no direction given; a bias-tuning baseline)
  rank1      one number per layer, shared by all steps, scaling the agreed rank-1 weight change
             (input-dependent, as in the adapters)

Training captions are COCO val2014 captions 1000-1199 of the fixed order; evaluation images
are written for captions 0-99, the same captions and seeds as study 7.

    python scripts/study/s10_train_shift.py [variant ...]
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
from adapters import SAMPLERS, bench, load_factors  # noqa: E402
from s2_function import HOSTS, consensus, random_direction  # noqa: E402
from s3_mechanism import coefficient, shaped  # noqa: E402
from s7_quality import coco_items  # noqa: E402

from wspd import config  # noqa: E402

STEPS, TEACHER_STEPS, GUIDANCE = 4, 20, 7.5
N_TRAIN, N_EVAL, EPOCHS = 200, 100, 2
LR = {"strength": 3e-2, "random": 3e-2, "bias": 1e-3, "rank1": 3e-2}
OUT = config.RUNS / "study" / "quality_trained"


def teacher_data(b, items) -> list[dict]:
    """Teacher states at the student's timesteps, and the final latent, per caption."""
    cache = config.RUNS / "study" / f"teacher_{len(items)}.pt"
    if cache.exists():
        return torch.load(cache)
    student = SAMPLERS["ddim"](b.sched_cfg)
    student.set_timesteps(STEPS)
    keep = {int(t) for t in student.timesteps}
    data = []
    for it in items:
        cond = b.encode(it["caption"])
        sched = SAMPLERS["ddim"](b.sched_cfg)
        sched.set_timesteps(TEACHER_STEPS, device=b.device)
        z = torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(it["image_id"])).to(b.device)
        states = {}
        for t in sched.timesteps:
            if int(t) in keep:
                states[int(t)] = z.cpu()
            eps_c, eps_u = b.unet(z, t, encoder_hidden_states=cond).sample, b.unet(z, t, encoder_hidden_states=b.empty).sample
            z = sched.step(eps_u + GUIDANCE * (eps_c - eps_u), t, z, return_dict=False)[0]
        assert len(states) == STEPS, "the teacher grid must contain the student timesteps"
        data.append({"cond": cond.cpu(), "states": [states[int(t)] for t in student.timesteps] + [z.cpu()]})
        if len(data) % 20 == 0:
            print(f"teacher {len(data)}/{len(items)}", flush=True)
    torch.save(data, cache)
    return data


def main() -> None:
    variants = sys.argv[1:] or ["untrained", "strength", "random", "bias", "rank1"]
    b = bench()
    device = b.device
    b.unet.requires_grad_(False)
    items = coco_items(1000 + N_TRAIN)
    train, evaluation = teacher_data(b, items[1000:]), items[:N_EVAL]
    factors = {h: load_factors(h) for h in HOSTS}
    layers = sorted(factors["lcm"])
    agreed = consensus([factors[h] for h in HOSTS], factors["lcm"])  # unit directions; the strength is not used
    rand = random_direction(factors["lcm"])
    mods = {n: b.unet.get_submodule(n) for n in layers}
    sched = SAMPLERS["ddim"](b.sched_cfg)
    sched.set_timesteps(STEPS, device=device)
    timesteps = sched.timesteps
    alphas = sched.alphas_cumprod.to(device)

    def move(z, eps, k):
        """One deterministic DDIM step from student timestep k to k+1 (or to the clean latent)."""
        a_t = alphas[timesteps[k]]
        a_next = alphas[timesteps[k + 1]] if k + 1 < STEPS else torch.tensor(1.0, device=device)
        x0 = (z - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        return a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps

    path = config.RESULTS / "study" / "s10_train.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    state = {"step": 0}
    for variant in variants:
        torch.manual_seed(0)
        if variant in ("strength", "random"):
            source = agreed if variant == "strength" else rand
            u = {n: shaped(mods[n], source[n][0][:, 0].to(device)) for n in layers}
            params = {n: torch.zeros(STEPS, device=device, requires_grad=True) for n in layers}
            hook = lambda n: (lambda m, a, out: out + u[n] * params[n][state["step"]])  # noqa: E731
        elif variant == "bias":
            params = {n: torch.zeros(STEPS, mods[n].weight.shape[0], device=device, requires_grad=True) for n in layers}
            hook = lambda n: (lambda m, a, out: out + shaped(m, params[n][state["step"]]))  # noqa: E731
        elif variant == "rank1":
            u = {n: shaped(mods[n], agreed[n][0][:, 0].to(device)) for n in layers}
            v = {n: agreed[n][2][0].to(device) for n in layers}
            params = {n: torch.zeros((), device=device, requires_grad=True) for n in layers}
            hook = lambda n: (lambda m, a, out: out + u[n] * (params[n] * coefficient(m, a[0], v[n])))  # noqa: E731
        elif variant == "untrained":  # the same sampling code with nothing added, as the floor
            params, hook = {}, None
        else:
            raise SystemExit(f"unknown variant {variant}")
        handles = [mods[n].register_forward_hook(hook(n)) for n in layers] if hook else []
        opt = torch.optim.Adam(params.values(), lr=LR[variant]) if params else None
        order = np.random.default_rng(0)
        losses, t0 = [], time.time()
        try:
            for epoch in range(EPOCHS if params else 0):
                for count, j in enumerate(order.permutation(len(train))):
                    ex = train[j]
                    cond = ex["cond"].to(device)
                    total = 0.0
                    for k in range(STEPS):
                        state["step"] = k
                        z, target = ex["states"][k].to(device), ex["states"][k + 1].to(device)
                        with torch.enable_grad():
                            loss = (move(z, b.unet(z, timesteps[k], encoder_hidden_states=cond).sample, k) - target).pow(2).mean()
                            opt.zero_grad()
                            loss.backward()
                        opt.step()
                        total += float(loss)
                    losses.append(total / STEPS)
                    if (count + 1) % 50 == 0:
                        print(f"{variant:9s} epoch {epoch} {count + 1:4d}/{len(train)} | loss {np.mean(losses[-50:]):.5f} | {time.time() - t0:5.0f} s", flush=True)
            # Evaluation images: 4 DDIM steps, no guidance, the study 7 captions and seeds.
            folder = OUT / f"trained_{variant}"
            folder.mkdir(parents=True, exist_ok=True)
            for i, it in enumerate(evaluation):
                cond = b.encode(it["caption"])
                z = torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(it["image_id"])).to(device)
                for k in range(STEPS):
                    state["step"] = k
                    z = move(z, b.unet(z, timesteps[k], encoder_hidden_states=cond).sample, k)
                x = b.vae.decode(z / b.vae.config.scaling_factor).sample
                Image.fromarray(((x.clamp(-1, 1) + 1) * 127.5).round().permute(0, 2, 3, 1).byte().cpu().numpy()[0]).save(folder / f"{i:05d}.png")
        finally:
            for h in handles:
                h.remove()
        learned = {n: p.detach().cpu() for n, p in params.items()}
        torch.save(learned, config.RUNS / "study" / f"trained_{variant}.pt")
        results[variant] = {
            "parameters": int(sum(p.numel() for p in learned.values())),
            "train_captions": len(train), "epochs": EPOCHS, "lr": LR.get(variant), "seconds": time.time() - t0,
            "loss_first_50": float(np.mean(losses[:50] or [0])), "loss_last_50": float(np.mean(losses[-50:] or [0])),
            "loss_curve": [float(np.mean(losses[i : i + 25])) for i in range(0, len(losses), 25)],
        }
        print(f"{variant:9s} done | {results[variant]['parameters']} parameters | loss {results[variant]['loss_first_50']:.5f} -> {results[variant]['loss_last_50']:.5f}", flush=True)
        path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
