"""Study 14: does a plain few-step loss point along the direction the adapters share?

No adapter is used to find the direction. At the base weights, the gradient of study 10's
training loss (4 DDIM steps without guidance, each matched to the 20-step guided teacher's
next state) is averaged over the 200 training captions, for every layer the adapters change.
The top singular direction of the negative gradient is the best rank-1 first-order change to
that layer. It is compared with each adapter's top direction, with the direction the five
methods share, with the fully fine-tuned models and with the style adapters, using the
measures of study 1.

    python scripts/study/s14_gradient.py

The directions are stored in runs/study/factors_gradient.pt; study 10 can train strengths on
them (variants `gradient` and `gradrank1`).

A second pass takes the gradient for a vector added to each layer's output, which is what
study 10's free vector follows, to see why that vector did not reach the shared direction.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from adapters import ACCEL, STYLE, bench, load_factors  # noqa: E402
from s1_weights import compare  # noqa: E402
from s2_function import HOSTS, consensus  # noqa: E402
from s3_mechanism import shaped  # noqa: E402
from s7_quality import coco_items  # noqa: E402
from s10_train_shift import N_TRAIN, STEPS, stepper, teacher_data  # noqa: E402

from wspd import config  # noqa: E402

KEEP = 8
CACHE = config.RUNS / "study" / "factors_gradient.pt"
BIAS = config.RUNS / "study" / "gradient_bias.pt"
HALVES = config.RUNS / "study" / "factors_gradient_halves.pt"
_loaded = []


def model():
    """The sampling bench, loaded once however many passes need it."""
    if not _loaded:
        _loaded.append(bench())
    return _loaded[0]
FULL = ["dmd2_full", "reflow2_full", "lcm_full"]  # study 9's factors, used when they are still cached


def top_factors(g: torch.Tensor):
    """Top singular triples of a layer's matrix through its smaller Gram matrix, and the top one's energy share."""
    g = g.flatten(1).double()
    wide = g.shape[0] <= g.shape[1]
    vals, vecs = torch.linalg.eigh(g @ g.T if wide else g.T @ g)
    vals, vecs = vals.flip(0).clamp_min(0), vecs.flip(1)
    s, q = vals[:KEEP].sqrt(), vecs[:, :KEEP]
    u, vh = (q, (q.T @ g) / s[:, None]) if wide else ((g @ q) / s, q.T)
    return (u.float(), s.float(), vh.float()), float(vals[0] / vals.sum())


def gradient() -> dict:
    """Per layer, the factors of minus the loss gradient: for each student step and for the sum over steps."""
    if CACHE.exists():
        return torch.load(CACHE)
    b = model()
    device = b.device
    train = teacher_data(b, coco_items(1000 + N_TRAIN)[1000:])
    weights = {n: b.unet.get_submodule(n).weight for n in sorted(load_factors("lcm"))}
    b.unet.requires_grad_(False)
    for w in weights.values():
        w.requires_grad_(True)
    timesteps, move = stepper(b)
    total = {n: torch.zeros(w.shape) for n, w in weights.items()}
    out = {"steps": [], "share_steps": [], "loss": []}
    t0 = time.time()
    for k in range(STEPS):
        loss_sum = 0.0
        for ex in train:
            z, target = ex["states"][k].to(device), ex["states"][k + 1].to(device)
            with torch.enable_grad():
                loss = (move(z, b.unet(z, timesteps[k], encoder_hidden_states=ex["cond"].to(device)).sample, k) - target).pow(2).mean()
                loss.backward()  # gradients add up over the captions
            loss_sum += float(loss)
        per, share = {}, {}
        for n, w in weights.items():
            g = w.grad.detach().cpu() / len(train)
            w.grad = None
            total[n] += g
            per[n], share[n] = top_factors(-g)
        out["steps"].append(per)
        out["share_steps"].append(share)
        out["loss"].append(loss_sum / len(train))
        print(f"step {k} | loss {out['loss'][-1]:.5f} | {time.time() - t0:5.0f} s", flush=True)
    both = {n: top_factors(-g) for n, g in total.items()}
    out["sum"], out["share_sum"] = {n: v[0] for n, v in both.items()}, {n: v[1] for n, v in both.items()}
    torch.save(out, CACHE)
    return out


def gradient_halves() -> list[dict]:
    """The same direction (summed over steps) from the even and from the odd training captions,
    to see how much of it is noise from the choice of captions."""
    if HALVES.exists():
        return torch.load(HALVES)
    b = model()
    device = b.device
    train = teacher_data(b, coco_items(1000 + N_TRAIN)[1000:])
    weights = {n: b.unet.get_submodule(n).weight for n in sorted(load_factors("lcm"))}
    b.unet.requires_grad_(False)
    for w in weights.values():
        w.requires_grad_(True)
    timesteps, move = stepper(b)
    out, t0 = [], time.time()
    for half in (0, 1):
        for k in range(STEPS):
            for ex in train[half::2]:
                z, target = ex["states"][k].to(device), ex["states"][k + 1].to(device)
                with torch.enable_grad():
                    (move(z, b.unet(z, timesteps[k], encoder_hidden_states=ex["cond"].to(device)).sample, k) - target).pow(2).mean().backward()
        out.append({n: top_factors(-w.grad.detach().cpu())[0] for n, w in weights.items()})
        for w in weights.values():
            w.grad = None
        print(f"half {half} | {time.time() - t0:5.0f} s", flush=True)
    b.unet.requires_grad_(False)
    torch.save(out, HALVES)
    return out


def bias_gradient() -> list[dict]:
    """Per student step and layer, the loss gradient for a vector added to the layer's output."""
    if BIAS.exists():
        return torch.load(BIAS)
    b = model()
    device = b.device
    train = teacher_data(b, coco_items(1000 + N_TRAIN)[1000:])
    mods = {n: b.unet.get_submodule(n) for n in sorted(load_factors("lcm"))}
    b.unet.requires_grad_(False)
    params = {n: torch.zeros(m.weight.shape[0], device=device, requires_grad=True) for n, m in mods.items()}
    handles = [m.register_forward_hook(lambda m, a, out, n=n: out + shaped(m, params[n])) for n, m in mods.items()]
    timesteps, move = stepper(b)
    out, t0 = [], time.time()
    try:
        for k in range(STEPS):
            for ex in train:
                z, target = ex["states"][k].to(device), ex["states"][k + 1].to(device)
                with torch.enable_grad():
                    (move(z, b.unet(z, timesteps[k], encoder_hidden_states=ex["cond"].to(device)).sample, k) - target).pow(2).mean().backward()
            out.append({n: (p.grad / len(train)).cpu().clone() for n, p in params.items()})
            for p in params.values():
                p.grad = None
            print(f"output-vector gradient, step {k} | {time.time() - t0:5.0f} s", flush=True)
    finally:
        for h in handles:
            h.remove()
    torch.save(out, BIAS)
    return out


def vector_agreement(vectors: dict, directions: dict) -> dict:
    """|cos| between a vector per layer and a unit direction per layer: weighted by the vector's energy
    (study 10's measure), the median over layers, and what random vectors would give."""
    cos = np.array([abs(float(torch.nn.functional.cosine_similarity(v, directions[n], dim=0))) for n, v in vectors.items()])
    w = np.array([float(v.pow(2).sum()) for v in vectors.values()])
    rand = np.array([math.sqrt(2 / (math.pi * len(v))) for v in vectors.values()])
    return {"weighted": float(cos @ w / w.sum()), "median": float(np.median(cos)), "random": float(rand @ w / w.sum())}


def unit(f):
    """The same directions with every layer's strengths scaled to norm 1, so layers are weighted by the other side only."""
    return {n: (u, s / s.norm(), vh) for n, (u, s, vh) in f.items()}


def main() -> None:
    g = gradient()
    factors = {k: load_factors(k) for k in [*ACCEL, *STYLE]}
    for key in FULL:
        path = config.RUNS / "study" / f"factors_{key}.pt"
        if path.exists():
            factors[key] = {n: v[:3] for n, v in torch.load(path).items()}
    targets = {"shared": consensus([factors[h] for h in HOSTS], factors["lcm"]), **factors}
    keep = ("layers", "out_cos_weighted", "out_cos_median", "in_cos_weighted", "random_out_cos", "mismatched_out_cos_weighted", "same_sign_share", "delta_cos_global", "out_topk_weighted", "random_topk", "by_block", "by_kind")
    results = {
        "train_captions": N_TRAIN,
        "loss_per_step": g["loss"],
        "gradient_top1_energy_share_median": {"sum": float(np.median(list(g["share_sum"].values()))), "steps": [float(np.median(list(s.values()))) for s in g["share_steps"]]},
        "pairs": {}, "pairs_weighted_by_gradient_too": {}, "steps_vs_shared": [], "steps_vs_each_other": {},
    }
    for name, f in targets.items():
        results["pairs"][name] = r = {k: v for k, v in compare(unit(g["sum"]), f).items() if k in keep}
        raw = compare(g["sum"], f)
        results["pairs_weighted_by_gradient_too"][name] = {k: raw[k] for k in ("out_cos_weighted", "in_cos_weighted", "delta_cos_global")}
        print(f"gradient vs {name:14s} out {r['out_cos_weighted']:.3f} (median {r['out_cos_median']:.3f}, random {r['random_out_cos']:.3f}, mismatched {r['mismatched_out_cos_weighted']:.3f}) | in {r['in_cos_weighted']:.3f} | same sign {r['same_sign_share']:.2f} | delta {r['delta_cos_global']:+.3f} | also weighted by gradient: out {raw['out_cos_weighted']:.3f}", flush=True)
    for k, per in enumerate(g["steps"]):
        r = compare(unit(per), targets["shared"])
        results["steps_vs_shared"].append({key: r[key] for key in ("out_cos_weighted", "out_cos_median", "in_cos_weighted", "same_sign_share")})
        print(f"step {k} gradient vs shared: out {r['out_cos_weighted']:.3f} (median {r['out_cos_median']:.3f}) | in {r['in_cos_weighted']:.3f}", flush=True)
    for a in range(STEPS):
        for c in range(a + 1, STEPS):
            results["steps_vs_each_other"][f"{a}|{c}"] = r = compare(unit(g["steps"][a]), unit(g["steps"][c]))["out_cos_weighted"]
            print(f"step {a} vs step {c} gradient: out {r:.3f}", flush=True)
    by = results["pairs"]["shared"]
    print("by block:", {k: round(v["out_cos_weighted"], 3) for k, v in by["by_block"].items()}, "| by layer type:", {k: round(v["out_cos_weighted"], 3) for k, v in by["by_kind"].items()}, flush=True)

    # The gradient for an added output vector, against the shared direction and the weight gradient's direction.
    bias = bias_gradient()
    shared_u = {n: v[0][:, 0] for n, v in targets["shared"].items()}
    weight_u = {n: v[0][:, 0] for n, v in g["sum"].items()}
    summed = {n: sum(step[n] for step in bias) for n in bias[0]}
    rows = {"sum_over_steps": summed, **{f"step_{k}": step for k, step in enumerate(bias)}}
    results["output_vector_gradient"] = out = {"vs_shared": {}, "sign_only_vs_shared": {}, "vs_weight_gradient_direction": {}}
    for name, vec in rows.items():
        out["vs_shared"][name] = vector_agreement(vec, shared_u)
        # Adam's first update follows the sign of the gradient in every coordinate.
        out["sign_only_vs_shared"][name] = vector_agreement({n: v.sign() for n, v in vec.items()}, shared_u)
        out["vs_weight_gradient_direction"][name] = vector_agreement(vec, weight_u)
        print(f"output-vector gradient {name:14s} vs shared {out['vs_shared'][name]['weighted']:.3f} (median {out['vs_shared'][name]['median']:.3f}, random {out['vs_shared'][name]['random']:.3f}) | sign only {out['sign_only_vs_shared'][name]['weighted']:.3f} (median {out['sign_only_vs_shared'][name]['median']:.3f}) | vs weight-gradient direction {out['vs_weight_gradient_direction'][name]['weighted']:.3f} (median {out['vs_weight_gradient_direction'][name]['median']:.3f})", flush=True)
    for tag in ("", "_e8"):
        path = config.RUNS / "study" / f"trained_bias{tag}.pt"
        if path.exists():
            trained = torch.load(path)
            per_step = [vector_agreement({n: -trained[n][k] for n in trained}, {n: torch.nn.functional.normalize(bias[k][n], dim=0) for n in trained}) for k in range(STEPS)]
            out[f"trained_vector{tag}_vs_its_gradient"] = per_step
            out[f"trained_vector{tag}_vs_weight_gradient_direction"] = [vector_agreement({n: trained[n][k] for n in trained}, weight_u) for k in range(STEPS)]
            print(f"trained free vector{tag}: |cos| with its own step's gradient {[round(r['weighted'], 3) for r in per_step]} | with the weight-gradient direction {[round(r['weighted'], 3) for r in out[f'trained_vector{tag}_vs_weight_gradient_direction']]}", flush=True)

    # How stable is the direction under the choice of captions? Layers count equally here.
    halves = [unit(h) for h in gradient_halves()]
    r = compare(halves[0], halves[1])
    results["split_half"] = {
        "even_vs_odd_captions": {k: r[k] for k in ("out_cos_weighted", "out_cos_median", "in_cos_weighted", "same_sign_share")},
        "each_half_vs_shared": [compare(h, targets["shared"])["out_cos_weighted"] for h in halves],
        "each_half_vs_all_captions": [compare(h, unit(g["sum"]))["out_cos_weighted"] for h in halves],
    }
    print(f"gradient from even against odd captions: out {r['out_cos_weighted']:.3f} (median {r['out_cos_median']:.3f}) | in {r['in_cos_weighted']:.3f} | each half vs shared {[round(x, 3) for x in results['split_half']['each_half_vs_shared']]} | each half vs all {[round(x, 3) for x in results['split_half']['each_half_vs_all_captions']]}", flush=True)
    (config.RESULTS / "study" / "s14_gradient.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
