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
  gradient   as strength, with the direction taken from the loss gradient at the base weights
             (study 14), so no existing adapter is used anywhere
  gradrank1  as rank1, with the gradient's rank-1 direction
  bias_adam  the free vector again, with the learning rate chosen as below
  bias_row   the free vector with one second moment per layer and step in place of Adam's one per
             coordinate, so that an update keeps the direction of the gradient
  bias_sgd   the free vector with plain gradient descent

  strength_lr, random_lr, gradient_lr   as strength, random and gradient, with the learning rate
             chosen as below

For the last six the learning rate is the one, of a short list, that gives the lowest training
loss after one pass over the first 100 training captions. While the best value is at an end of the
list, the list is extended on that side. S10_GRID=a,b,c replaces the starting list.

Training captions are COCO val2014 captions 1000-1199 of the fixed order; evaluation images
are written for captions 0-99, the same captions and seeds as study 7.

    [S10_EPOCHS=8] python scripts/study/s10_train_shift.py [variant ...]

For the free vector, its agreement with the adapters' shared direction is recorded after every epoch.
"""
from __future__ import annotations

import json
import os
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
from wspd.store import device_lock  # noqa: E402

STEPS, TEACHER_STEPS, GUIDANCE = 4, 20, 7.5
N_TRAIN, N_EVAL = 200, 100
EPOCHS = int(os.environ.get("S10_EPOCHS", 2))  # runs with another length are stored under <variant>_e<epochs>
TAG = "" if EPOCHS == 2 else f"_e{EPOCHS}"
LR = {"strength": 3e-2, "random": 3e-2, "bias": 1e-3, "rank1": 3e-2, "gradient": 3e-2, "gradrank1": 3e-2}
GRID = {"bias_adam": (3e-4, 1e-3, 3e-3), "bias_row": (1e-4, 3e-4, 1e-3), "bias_sgd": (3.0, 10.0, 30.0, 100.0, 300.0)}
GRID |= {f"{v}_lr": (1e-2, 3e-2, 1e-1) for v in ("strength", "random", "gradient")}
SEARCH_CAPTIONS = 100
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


def stepper(b):
    """The student's timesteps, and its deterministic DDIM step from timestep k to k+1 (or to the clean latent)."""
    sched = SAMPLERS["ddim"](b.sched_cfg)
    sched.set_timesteps(STEPS, device=b.device)
    timesteps, alphas = sched.timesteps, sched.alphas_cumprod.to(b.device)

    def move(z, eps, k):
        a_t = alphas[timesteps[k]]
        a_next = alphas[timesteps[k + 1]] if k + 1 < STEPS else torch.tensor(1.0, device=b.device)
        x0 = (z - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        return a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps

    return timesteps, move


def neighbour(a: float, up: bool) -> float:
    """The next value in the series 1, 3, 10, 30, ..."""
    mantissa, exponent = f"{a:e}".split("e")
    three, e = mantissa.startswith("3"), int(exponent)
    if up:
        return float(f"1e{e + 1}") if three else float(f"3e{e}")
    return float(f"1e{e}") if three else float(f"3e{e - 1}")


class RowAdam(torch.optim.Optimizer):
    """Adam with one second moment per row of each parameter (a layer at a step) in place of one per
    coordinate. Every coordinate of a row is divided by the same number, so the update keeps the
    direction of the gradient, and rows with small gradients still move as in Adam."""

    def __init__(self, params, lr: float, betas=(0.9, 0.999), eps: float = 1e-8):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps})

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st.update(t=0, m=torch.zeros_like(p), v=torch.zeros(p.shape[0], 1, device=p.device))
                st["t"] += 1
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["v"].mul_(b2).add_(p.grad.pow(2).mean(1, keepdim=True), alpha=1 - b2)
                p.addcdiv_(st["m"] / (1 - b1 ** st["t"]), (st["v"] / (1 - b2 ** st["t"])).sqrt() + group["eps"], value=-group["lr"])


def optimizer(variant: str, params, lr: float):
    if variant == "bias_sgd":
        return torch.optim.SGD(params, lr=lr)
    if variant == "bias_row":
        return RowAdam(params, lr=lr)
    return torch.optim.Adam(params, lr=lr)


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
    timesteps, move = stepper(b)
    grad = torch.load(config.RUNS / "study" / "factors_gradient.pt")["sum"] if any(v.startswith("grad") for v in variants) else None

    path = config.RESULTS / "study" / "s10_train.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    state = {"step": 0}
    for variant in variants:
        torch.manual_seed(0)
        if variant.removesuffix("_lr") in ("strength", "random", "gradient"):
            source = {"strength": agreed, "random": rand, "gradient": grad}[variant.removesuffix("_lr")]
            u = {n: shaped(mods[n], source[n][0][:, 0].to(device)) for n in layers}
            params = {n: torch.zeros(STEPS, device=device, requires_grad=True) for n in layers}
            hook = lambda n: (lambda m, a, out: out + u[n] * params[n][state["step"]])  # noqa: E731
        elif variant.startswith("bias"):
            params = {n: torch.zeros(STEPS, mods[n].weight.shape[0], device=device, requires_grad=True) for n in layers}
            hook = lambda n: (lambda m, a, out: out + shaped(m, params[n][state["step"]]))  # noqa: E731
        elif variant in ("rank1", "gradrank1"):
            source = agreed if variant == "rank1" else grad
            u = {n: shaped(mods[n], source[n][0][:, 0].to(device)) for n in layers}
            v = {n: source[n][2][0].to(device) for n in layers}
            params = {n: torch.zeros((), device=device, requires_grad=True) for n in layers}
            hook = lambda n: (lambda m, a, out: out + u[n] * (params[n] * coefficient(m, a[0], v[n])))  # noqa: E731
        elif variant == "untrained":  # the same sampling code with nothing added, as the floor
            params, hook = {}, None
        else:
            raise SystemExit(f"unknown variant {variant}")
        handles = [mods[n].register_forward_hook(hook(n)) for n in layers] if hook else []
        alignment, t0 = [], time.time()

        def shared_agreement() -> float:
            """Has the free vector moved toward the direction the adapters share? Weighted by its energy."""
            num = den = 0.0
            for n in layers:
                for k in range(STEPS):
                    q = params[n][k].detach().cpu()
                    num += float(torch.nn.functional.cosine_similarity(q, agreed[n][0][:, 0], dim=0).abs() * q.pow(2).sum())
                    den += float(q.pow(2).sum())
            return num / max(den, 1e-30)

        def fit(lr: float, examples: list, epochs: int, log: bool) -> list:
            """Trains from zero and returns the loss per caption."""
            for q in params.values():
                q.zero_()
            opt, order, losses = optimizer(variant, params.values(), lr), np.random.default_rng(0), []
            for epoch in range(epochs):
                for count, j in enumerate(order.permutation(len(examples))):
                    ex = examples[j]
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
                    if log and (count + 1) % 50 == 0:
                        print(f"{variant:9s} epoch {epoch} {count + 1:4d}/{len(examples)} | loss {np.mean(losses[-50:]):.5f} | {time.time() - t0:5.0f} s", flush=True)
                if log and variant.startswith("bias"):
                    alignment.append(shared_agreement())
                    print(f"{variant} epoch {epoch} | agreement with the shared direction {alignment[-1]:.3f}", flush=True)
            return losses

        try:
            lr, search = LR.get(variant), {}
            if variant in GRID:
                grid = sorted(float(a) for a in os.environ["S10_GRID"].split(",")) if os.environ.get("S10_GRID") else sorted(GRID[variant])
                todo = list(grid)
                while todo:
                    a = todo.pop(0)
                    tail = float(np.mean(fit(a, train[:SEARCH_CAPTIONS], 1, False)[-50:]))
                    search[str(a)] = tail if np.isfinite(tail) else None
                    print(f"{variant:9s} learning rate {a:g} | loss on the last 50 of {SEARCH_CAPTIONS} captions {tail:.5f} | {time.time() - t0:5.0f} s", flush=True)
                    if todo:
                        continue
                    lr = min((a for a in grid if search[str(a)] is not None), key=lambda a: search[str(a)])
                    if lr in (grid[0], grid[-1]) and len(grid) < 9:  # the best value is at an end: look one further
                        todo.append(neighbour(lr, up=lr == grid[-1]))
                        grid = sorted(grid + todo)
                print(f"{variant:9s} learning rate chosen: {lr:g}", flush=True)
            losses = fit(lr, train, EPOCHS, True) if params else []
            # Evaluation images: 4 DDIM steps, no guidance, the study 7 captions and seeds.
            folder = OUT / f"trained_{variant}{TAG}"
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
        torch.save(learned, config.RUNS / "study" / f"trained_{variant}{TAG}.pt")
        with_gradient = []
        bias_gradient = config.RUNS / "study" / "gradient_bias.pt"  # written by study 14
        if variant.startswith("bias") and bias_gradient.exists():
            for k, g in enumerate(torch.load(bias_gradient)):
                energy = {n: float(learned[n][k].pow(2).sum()) for n in layers}
                cosine = {n: float(torch.nn.functional.cosine_similarity(learned[n][k], -g[n], dim=0)) for n in layers}
                with_gradient.append(sum(cosine[n] * energy[n] for n in layers) / max(sum(energy.values()), 1e-30))
        results[variant + TAG] = {
            "parameters": int(sum(p.numel() for p in learned.values())),
            "train_captions": len(train), "epochs": EPOCHS, "lr": lr, "lr_search": search, "seconds": time.time() - t0,
            "cosine_with_minus_loss_gradient_per_step": with_gradient,
            "loss_first_50": float(np.mean(losses[:50] or [0])), "loss_last_50": float(np.mean(losses[-50:] or [0])),
            "loss_curve": [float(np.mean(losses[i : i + 25])) for i in range(0, len(losses), 25)],
            "agreement_with_shared_direction_per_epoch": alignment,
        }
        r = results[variant + TAG]
        print(f"{variant:9s} done | {r['parameters']} parameters | loss {r['loss_first_50']:.5f} -> {r['loss_last_50']:.5f}", flush=True)
        path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    with device_lock():
        main()
