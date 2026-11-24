"""Study 18: does what was found on the base model carry over to fine-tuned models?

Public adapters are trained on the base model and then used on community fine-tunes of it. Here
the reduced forms of the earlier studies are applied to a fine-tune in the same way. Nothing is
fitted or trained on the fine-tune. Arms, for the first COCO captions of study 7 with the same
seeds, 4 steps without guidance unless stated:
  ref_base25          the fine-tune itself, 25 steps, guidance 7.5
  base4_guided        the fine-tune, 4 steps, guidance 7.5
  none_<sampler>      the fine-tune without an adapter, with each adapter's sampler
  <host>_full         the adapter as released
  <host>_rank1        its top direction per layer
  <host>_const        one fixed vector per layer, with the constants fitted on the base model (study 7)
  <host>_loo          the adapter built from the other four methods, with the scale factor chosen on
                      the base model (study 13)
  trained_strength    the shared direction with the strengths trained on the base model (study 10)
  trained_gradient    the base model's loss-gradient direction with the strengths trained on the
                      base model (studies 14 and 10); no public adapter is involved

    python scripts/study/s18_finetunes.py <model> [n_captions] [block]   # images; <model> is ds8 or rv51
    python scripts/study/s18_finetunes.py distance <model>               # how far its UNet is from the base
    python scripts/study/s7_score.py --no-fid --ft_<model>               # scores
    python scripts/study/s18_finetunes.py table                          # results/study/s18_finetunes.json
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
from adapters import ADAPTERS, BASE, BASE_REV, SAMPLERS, bench, load_factors, rank1  # noqa: E402
from s2_function import HOSTS, consensus  # noqa: E402
from s3_mechanism import shaped  # noqa: E402
from s7_quality import STEPS, coco_items  # noqa: E402
from s10_train_shift import stepper  # noqa: E402
from s12_ablation import sample  # noqa: E402
from s13_transfer import leave_one_out, scaled  # noqa: E402

from wspd import config  # noqa: E402
from wspd.metrics import paired_bootstrap  # noqa: E402
from wspd.model import best_device  # noqa: E402
from wspd.store import device_lock  # noqa: E402

# name -> (repository, revision, weight-file variant)
FINETUNES = {
    "ds8": ("Lykon/dreamshaper-8", "a7e52b98680b1ba8ff7bce97c7f9f2e2e5337917", "fp16"),
    "rv51": ("SG161222/Realistic_Vision_V5.1_noVAE", "1e9f017a7b1eaefb63a1900ea6c5953d2739fd21", None),
}
STUDY = config.RUNS / "study"
FEATS = STUDY / "quality_features"


def root(tag: str) -> Path:
    return STUDY / f"quality_ft_{tag}"


def guided(b, item: dict, steps: int, guidance: float = 7.5) -> np.ndarray:
    """A DDIM image with guidance, as in study 7."""
    cond = b.encode(item["caption"])
    sched = SAMPLERS["ddim"](b.sched_cfg)
    sched.set_timesteps(steps, device=b.device)
    z = torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(item["image_id"])).to(b.device) * sched.init_noise_sigma
    for t in sched.timesteps:
        zin = sched.scale_model_input(z, t)
        eps, eps_un = b.unet(zin, t, encoder_hidden_states=cond).sample, b.unet(zin, t, encoder_hidden_states=b.empty).sample
        z = sched.step(eps_un + guidance * (eps - eps_un), t, z, return_dict=False)[0]
    x = b.vae.decode(z / b.vae.config.scaling_factor).sample
    return ((x.clamp(-1, 1) + 1) * 127.5).round().permute(0, 2, 3, 1).byte().cpu().numpy()[0]


def generate(tag: str, n: int, size: int) -> None:
    repo, rev, variant = FINETUNES[tag]
    items = coco_items(n)
    b = bench(model=repo, revision=rev, variant=variant)
    device = b.device
    factors = {h: load_factors(h) for h in HOSTS}
    consts = {h: torch.load(STUDY / f"constants_{h}.pt") for h in HOSTS}  # fitted on the base model
    chosen = json.loads((config.RESULTS / "study" / "s13_transfer.json").read_text())["dev"]  # chosen on the base model
    directions = {"strength": consensus([factors[h] for h in HOSTS], factors["lcm"]), "gradient": torch.load(STUDY / "factors_gradient.pt")["sum"]}
    timesteps, move = stepper(b)
    state = {"step": 0}

    def constant_hooks(h: str) -> list:
        handles = []
        for name, (u, s, _) in factors[h].items():
            module = b.unet.get_submodule(name)
            shift = shaped(module, u[:, 0].to(device)) * (float(s[0]) * float(np.mean(consts[h][name])))
            handles.append(module.register_forward_hook(lambda m, a, out, shift=shift: out + shift))
        return handles

    def strength_hooks(kind: str) -> list:
        learned = torch.load(STUDY / f"trained_{kind}.pt")
        handles = []
        for name, p in learned.items():
            module = b.unet.get_submodule(name)
            u, p = shaped(module, directions[kind][name][0][:, 0].to(device)), p.to(device)
            handles.append(module.register_forward_hook(lambda m, a, out, u=u, p=p: out + u * p[state["step"]]))
        return handles

    def trained_image(item: dict) -> np.ndarray:
        """Study 10's sampler: 4 deterministic DDIM steps."""
        cond = b.encode(item["caption"])
        z = torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(item["image_id"])).to(device)
        for k in range(STEPS):
            state["step"] = k
            z = move(z, b.unet(z, timesteps[k], encoder_hidden_states=cond).sample, k)
        x = b.vae.decode(z / b.vae.config.scaling_factor).sample
        return ((x.clamp(-1, 1) + 1) * 127.5).round().permute(0, 2, 3, 1).byte().cpu().numpy()[0]

    # name -> (weight change, hooks, image function)
    arms = {
        "ref_base25": (None, None, lambda it: guided(b, it, 25)),
        "base4_guided": (None, None, lambda it: guided(b, it, STEPS)),
    }
    for sampler in sorted({ADAPTERS[h].sampler for h in HOSTS}):
        arms[f"none_{sampler}"] = (None, None, lambda it, sampler=sampler: sample(b, it, sampler))
    for h in HOSTS:
        draw = lambda it, h=h: sample(b, it, ADAPTERS[h].sampler)  # noqa: E731
        arms[f"{h}_full"] = (factors[h], None, draw)
        arms[f"{h}_rank1"] = (rank1(factors[h]), None, draw)
        arms[f"{h}_const"] = (None, lambda h=h: constant_hooks(h), draw)
        arms[f"{h}_loo"] = (scaled(leave_one_out([factors[k] for k in HOSTS if k != h]), chosen[h]["loo_factor"]), None, draw)
    for kind in ("strength", "gradient"):
        arms[f"trained_{kind}"] = (None, lambda kind=kind: strength_hooks(kind), trained_image)

    for start in range(0, n, size):
        for name, (change, hooks, draw) in arms.items():
            folder = root(tag) / name
            folder.mkdir(parents=True, exist_ok=True)
            todo = [(i, it) for i, it in enumerate(items[start : start + size], start) if not (folder / f"{i:05d}.png").exists()]
            if not todo:
                continue
            t0 = time.time()
            b.install(change)
            handles = hooks() if hooks else []
            try:
                for i, it in todo:
                    Image.fromarray(draw(it)).save(folder / f"{i:05d}.png")
            finally:
                for handle in handles:
                    handle.remove()
            print(f"{tag} block {start:4d} | {name:17s} | {len(todo)} images | {time.time() - t0:5.0f} s", flush=True)
    b.install(None)


def distance(tag: str) -> dict:
    """How far the fine-tune's UNet is from the base model's, over the layers the adapters change.
    Stored, so that the table can be rebuilt after the checkpoint has been removed from disk."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = config.RESULTS / "study" / "s18_distance.json"
    known = json.loads(path.read_text()) if path.exists() else {}
    if tag in known:
        return known[tag]
    repo, rev, variant = FINETUNES[tag]
    file = f"unet/diffusion_pytorch_model{'.' + variant if variant else ''}.safetensors"
    lcm = load_factors("lcm")
    change = size = 0.0
    with safe_open(hf_hub_download(BASE, "unet/diffusion_pytorch_model.safetensors", revision=BASE_REV), "pt") as base, safe_open(hf_hub_download(repo, file, revision=rev), "pt") as tuned:
        for n in lcm:
            w0, w1 = base.get_tensor(f"{n}.weight").float(), tuned.get_tensor(f"{n}.weight").float()
            change += float((w1 - w0).pow(2).sum())
            size += float(w0.pow(2).sum())
    known[tag] = {"repository": repo, "revision": rev, "relative_change": (change / size) ** 0.5, "change_over_lcm_lora": (change / sum(float(s.pow(2).sum()) for _, s, _ in lcm.values())) ** 0.5}
    path.write_text(json.dumps(known, indent=1))
    return known[tag]


def table() -> None:
    """Gains over no adapter on each model, from the per-image PickScores that s7_score.py cached."""
    def pick(prefix: str, arm: str, n: int) -> np.ndarray:
        return np.load(FEATS / f"{prefix}pick__{arm}.npy")[:n]

    # Where each arm's base-model images were scored: study 7, study 13 (transfer_) and study 10.
    base_prefix = {"full": "", "rank1": "", "const": "", "loo": "transfer_"}
    out = {}
    for tag in FINETUNES:
        if not (FEATS / f"ft_{tag}_pick__lcm_full.npy").exists():
            continue
        n = len(np.load(FEATS / f"ft_{tag}_pick__lcm_full.npy"))
        model = {"captions": n, "distance_from_base": distance(tag), "references": {}, "arms": {}}
        for arm in ("ref_base25", "base4_guided"):
            model["references"][arm] = {"fine_tune": float(pick(f"ft_{tag}_", arm, n).mean()), "base": float(pick("", arm, n).mean())}
        rows = [(f"{h}_{v}", f"none_{ADAPTERS[h].sampler}", f"{h}_full", base_prefix[v]) for h in HOSTS for v in ("full", "rank1", "const", "loo")]
        rows += [(f"trained_{k}", "none_ddim", None, "") for k in ("strength", "gradient")]
        for arm, none, full, prefix in rows:
            r = {}
            for where, pre, arm_pre in (("fine_tune", f"ft_{tag}_", f"ft_{tag}_"), ("base", "", prefix)):
                a, floor = pick(arm_pre, arm, n), pick(pre, none, n)
                r[where] = {"pick": float(a.mean()), "none": float(floor.mean()), "gain_over_none": paired_bootstrap(a - floor)}
                if full and arm != full:
                    top = pick(pre, full, n)
                    r[where]["minus_full"] = paired_bootstrap(a - top)
                    r[where]["share_of_full_gain"] = float((a - floor).mean() / (top - floor).mean())
            model["arms"][arm] = r
        out[tag] = model
        d = model["distance_from_base"]
        print(f"\n{tag}: {n} captions | relative change of the UNet {d['relative_change']:.3f} | {d['change_over_lcm_lora']:.1f} times LCM-LoRA's change", flush=True)
        for arm, r in model["references"].items():
            print(f"  {arm:17s} fine-tune {r['fine_tune']:.2f} | base {r['base']:.2f}", flush=True)
        for arm, r in model["arms"].items():
            share = lambda w: f" ({r[w]['share_of_full_gain']:.0%} of full)" if "share_of_full_gain" in r[w] else ""  # noqa: E731
            print(f"  {arm:17s} fine-tune {r['fine_tune']['pick']:.2f}, gain {r['fine_tune']['gain_over_none']['mean']:+.2f}{share('fine_tune')} | base {r['base']['pick']:.2f}, gain {r['base']['gain_over_none']['mean']:+.2f}{share('base')}", flush=True)
    (config.RESULTS / "study" / "s18_finetunes.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "table":
        table()
    elif sys.argv[1] == "distance":
        print(distance(sys.argv[2]))
    else:
        run = lambda: generate(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 100, int(sys.argv[3]) if len(sys.argv) > 3 else 50)  # noqa: E731
        if best_device() == "cpu":  # a small job beside the GPU; nothing to queue for
            run()
        else:
            with device_lock():
                run()
