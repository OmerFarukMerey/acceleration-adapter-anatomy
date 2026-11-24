"""Study 17: which layer types can be replaced by a constant, and why Flash Diffusion cannot.

A rank-1 change adds `s * u * (v . x)` to a layer's output. Study 3 replaced the coefficient
`v . x` by its mean in every layer at once. Here the layers are split into groups: the time
projections, convolutions, the transformer blocks' input and output projections, feed-forward
layers, and the query, key, value and output projections of self- and cross-attention.

    python scripts/study/s17_layer_types.py measure
        How the coefficient varies, per group and adapter, on the 16 development prompts with
        the rank-1 adapter running. Its mean square is split into the part a constant per layer
        keeps and the parts it drops: variation between steps, between prompts, and between
        positions in the image. No image is scored. Into results/study/s17_layer_types.json.

    python scripts/study/s17_layer_types.py [n_captions]
        Images for the first COCO captions of study 7, same seeds, 4 steps without guidance.
        The constants are study 7's, fitted on the development prompts. Arms:
          <host>_const_<group>   the constant in one group of layers, the top direction elsewhere.
                                 Groups for LCM, TCD, PCM and Hyper-SD: conv, ff, attn (all eight
                                 attention projections), proj (time and block projections). For
                                 Flash Diffusion, which changes attention only: keys, queries,
                                 values, outs.
          <host>_const2          the constant in every layer, with two constants in the key and
                                 value layers of cross-attention, whose input is the prompt's 77
                                 tokens: one for the start token, one for the other 76
          flash_const2_values    the same two constants in Flash Diffusion's value layers, the top
                                 direction elsewhere
          flash_drop_keys        the top direction with the key layers left out
          flash_keep_keys        the constant everywhere except the key layers
          flash_posavg           every coefficient replaced by its average over positions, taken
                                 from the running model (study 3's arm)
          flash_posrec           the same averages, recorded per caption and step from a run with
                                 the top direction and then played back
        The full adapter, its top direction, the constant in every layer and the base model are
        linked from study 7.

    python scripts/study/s7_score.py --no-fid --kinds
    python scripts/study/s17_layer_types.py table      # differences against the top direction

    python scripts/study/s17_layer_types.py tokens [n_captions]
        The coefficient of each cross-attention key and value layer per token of the prompt, for
        the first COCO captions of study 7: the start token, the caption's words, and the end
        token with the padding after it. Text encoder only.

    python scripts/study/s17_layer_types.py attention
        How cross-attention divides its weight between the start token, the words, and the end
        token with the padding, on the development prompts with each adapter's top direction
        running.
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
from adapters import ADAPTERS, BASE, BASE_REV, bench, kind, load_factors, rank1  # noqa: E402
from s3_mechanism import coefficient, shaped  # noqa: E402
from s7_quality import ROOT as QUALITY  # noqa: E402
from s7_quality import STEPS, coco_items  # noqa: E402
from s12_ablation import sample  # noqa: E402

from wspd import config  # noqa: E402
from wspd.metrics import paired_bootstrap  # noqa: E402
from wspd.model import best_device  # noqa: E402
from wspd.store import device_lock  # noqa: E402

HOSTS = ["lcm", "tcd", "pcm", "hyper4", "flash"]
GROUPS = ("time", "conv", "attn_proj", "ff", "self_q", "self_k", "self_v", "self_out", "cross_q", "cross_k", "cross_v", "cross_out")
PARTS = ("constant", "between_steps", "between_prompts", "between_positions")
ROOT = config.RUNS / "study" / "quality_kinds"
RESULT = config.RESULTS / "study" / "s17_layer_types.json"


def group(name: str) -> str:
    k = kind(name)
    if k in ("self_attn", "cross_attn"):
        return f"{k.split('_')[0]}_{name.rsplit('.to_', 1)[1].split('.')[0]}"
    return k


def variation(b, host: str) -> dict:
    """Per layer: the coefficient's mean per step and prompt, and its mean square."""
    f = load_factors(host)
    mods = {n: b.unet.get_submodule(n) for n in f}
    state = {"step": 0, "prompt": 0}
    mean = {n: np.zeros((STEPS, len(b.prompts))) for n in f}
    square = {n: np.zeros((STEPS, len(b.prompts))) for n in f}

    def record(n, v):
        def hook(module, args, output):
            c = coefficient(module, args[0], v)
            mean[n][state["step"], state["prompt"]] = float(c.mean())
            square[n][state["step"], state["prompt"]] = float(c.pow(2).mean())
        return hook

    b.install(rank1(f))
    handles = [m.register_forward_hook(record(n, f[n][2][0].to(b.device))) for n, m in mods.items()]
    try:
        for _ in b.trajectory(ADAPTERS[host].sampler, STEPS, before=lambda i, step: state.update(step=step, prompt=i)):
            pass
    finally:
        for h in handles:
            h.remove()
        b.install(None)

    layers = {}
    for n in f:
        m, total = mean[n], float(square[n].mean())
        per_step = m.mean(1)
        split = [float(m.mean() ** 2), float(((per_step - m.mean()) ** 2).mean()), float(((m - per_step[:, None]) ** 2).mean()), total - float((m**2).mean())]
        strength = float(f[n][1][0] ** 2)
        layers[n] = {"weight_energy": strength, "output_energy": strength * total, **{p: v / total for p, v in zip(PARTS, split)}}

    def pooled(names: list) -> dict:
        out_energy = sum(layers[n]["output_energy"] for n in names)
        return {
            "layers": len(names),
            "share_of_weight_energy": sum(layers[n]["weight_energy"] for n in names) / sum(v["weight_energy"] for v in layers.values()),
            "share_of_output_energy": out_energy / sum(v["output_energy"] for v in layers.values()),
            **{p: sum(layers[n][p] * layers[n]["output_energy"] for n in names) / out_energy for p in PARTS},
        }

    attention = [n for n in layers if kind(n) in ("self_attn", "cross_attn")]
    return {"all": pooled(list(layers)), "attention_only": pooled(attention), "groups": {g: pooled([n for n in layers if group(n) == g]) for g in GROUPS if any(group(n) == g for n in layers)}}


def measure() -> None:
    b = bench()
    results = json.loads(RESULT.read_text()) if RESULT.exists() else {}
    results.setdefault("variation", {})
    for host in HOSTS:
        t0 = time.time()
        results["variation"][host] = r = variation(b, host)
        RESULT.write_text(json.dumps(results, indent=1))
        print(f"\n{host} ({time.time() - t0:.0f} s)   group: layers | weight energy | output energy | constant, steps, prompts, positions", flush=True)
        for name, g in [("all", r["all"]), ("attention", r["attention_only"]), *r["groups"].items()]:
            print(f"  {name:10s} {g['layers']:4d} | {g['share_of_weight_energy']:.3f} | {g['share_of_output_energy']:.3f} | " + ", ".join(f"{g[p]:.2f}" for p in PARTS), flush=True)


COARSE = {"conv": ("conv",), "ff": ("ff",), "attn": tuple(g for g in GROUPS if g.startswith(("self_", "cross_"))), "proj": ("time", "attn_proj")}
FLASH = {"keys": ("self_k", "cross_k"), "queries": ("self_q", "cross_q"), "values": ("self_v", "cross_v"), "outs": ("self_out", "cross_out")}


def images(n: int) -> None:
    items = coco_items(n)
    b = bench()
    device = b.device
    state = {"step": 0}
    track = lambda step: state.update(step=step)  # noqa: E731
    conds = b.conds  # the 16 development prompts
    arms = {}  # name -> (host, weight change, hooks or None, image function or None)

    for h in HOSTS:
        f = load_factors(h)
        top = rank1(f)
        const = {n: float(np.mean(v)) for n, v in torch.load(config.RUNS / "study" / f"constants_{h}.pt").items()}

        # In the key and value layers of cross-attention the input is the prompt's 77 token embeddings.
        # The first is the start token, whose embedding is the same for every prompt. Two constants
        # for such a layer: the start token's coefficient, and the mean over the other 76 tokens of
        # the development prompts.
        text = [n for n in f if n.endswith(("attn2.to_k", "attn2.to_v"))]
        tokens = {n: torch.stack([cond[0] @ f[n][2][0].to(device) for cond in conds]) for n in text}  # prompts x 77
        by_token = {n: torch.cat([c[:, :1].mean().view(1), c[:, 1:].mean().expand(c.shape[1] - 1)]) for n, c in tokens.items()}

        def constant_hooks(names, split=False, f=f, const=const, by_token=by_token) -> list:
            handles = []
            for name in names:
                module = b.unet.get_submodule(name)
                value = by_token[name][None, :, None] if split and name in by_token else const[name]
                shift = shaped(module, f[name][0][:, 0].to(device)) * (float(f[name][1][0]) * value)
                handles.append(module.register_forward_hook(lambda m, a, out, shift=shift: out + shift))
            return handles

        arms[f"{h}_const2"] = (h, None, lambda names=list(f), make=constant_hooks: make(names, split=True), None)

        for label, fine in (FLASH if h == "flash" else COARSE).items():
            inside = [n for n in f if group(n) in fine]
            arms[f"{h}_const_{label}"] = (h, {n: v for n, v in top.items() if n not in inside}, lambda inside=inside, make=constant_hooks: make(inside), None)
        if h != "flash":
            continue
        values = [n for n in f if group(n) in FLASH["values"]]
        arms["flash_const2_values"] = (h, {n: v for n, v in top.items() if n not in values}, lambda values=values, make=constant_hooks: make(values, split=True), None)
        keys = [n for n in f if group(n) in FLASH["keys"]]
        arms["flash_drop_keys"] = (h, {n: v for n, v in top.items() if n not in keys}, None, None)
        arms["flash_keep_keys"] = (h, {n: top[n] for n in keys}, lambda rest=[n for n in f if n not in keys], make=constant_hooks: make(rest), None)

        mods = {n: b.unet.get_submodule(n) for n in f}
        parts = {n: (shaped(mods[n], u[:, 0].to(device)), float(s[0]), vh[0].to(device)) for n, (u, s, vh) in f.items()}

        def average(module, x, v):
            c = coefficient(module, x, v)
            return c.mean(dim=tuple(range(1, c.dim())), keepdim=True)

        def live_hooks() -> list:
            return [m.register_forward_hook(lambda m, a, out, p=parts[n]: out + p[0] * (p[1] * average(m, a[0], p[2]))) for n, m in mods.items()]

        def played_back(item: dict, top=top) -> np.ndarray:
            seen = {}
            b.install(top)
            handles = [m.register_forward_hook(lambda m, a, out, n=n: seen.__setitem__((n, state["step"]), average(m, a[0], parts[n][2]))) for n, m in mods.items()]
            try:
                sample(b, item, ADAPTERS["flash"].sampler, before=track)
            finally:
                for handle in handles:
                    handle.remove()
            b.install(None)
            handles = [m.register_forward_hook(lambda m, a, out, n=n: out + parts[n][0] * (parts[n][1] * seen[n, state["step"]])) for n, m in mods.items()]
            try:
                return sample(b, item, ADAPTERS["flash"].sampler, before=track)
            finally:
                for handle in handles:
                    handle.remove()

        arms["flash_posavg"] = (h, None, live_hooks, None)
        arms["flash_posrec"] = (h, None, None, played_back)

    # The reference arms were generated by study 7 with the same captions, seeds and sampling code.
    for name in [f"{h}_{v}" for h in HOSTS for v in ("full", "rank1", "const")] + [f"none_{s}" for s in sorted({ADAPTERS[h].sampler for h in HOSTS})]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            link = ROOT / name / f"{i:05d}.png"
            if not link.exists():
                link.symlink_to(QUALITY / name / f"{i:05d}.png")

    for name, (h, change, hooks, draw) in arms.items():
        folder = ROOT / name
        folder.mkdir(parents=True, exist_ok=True)
        todo = [(i, it) for i, it in enumerate(items) if not (folder / f"{i:05d}.png").exists()]
        if not todo:
            continue
        t0 = time.time()
        b.install(change)
        handles = hooks() if hooks else []
        try:
            for i, it in todo:
                Image.fromarray(draw(it) if draw else sample(b, it, ADAPTERS[h].sampler)).save(folder / f"{i:05d}.png")
        finally:
            for handle in handles:
                handle.remove()
        b.install(None)
        print(f"{name:20s} | {len(change or {}):3d} layers with the top direction | {len(todo)} images | {time.time() - t0:4.0f} s", flush=True)


def table() -> None:
    """Per arm, PickScore against the same adapter's top direction, from the scores s7_score.py cached."""
    feats = config.RUNS / "study" / "quality_features"
    load = lambda what, arm: np.load(feats / f"kinds_{what}__{arm}.npy")  # noqa: E731
    results = json.loads(RESULT.read_text())
    results["quality"] = out = {}
    for h in HOSTS:
        top, floor, const = load("pick", f"{h}_rank1"), load("pick", f"none_{ADAPTERS[h].sampler}"), load("pick", f"{h}_const")
        out[h] = rows = {"captions": len(top), "rank1": float(top.mean()), "none": float(floor.mean()), "full": float(load("pick", f"{h}_full").mean())}
        for path in sorted(feats.glob(f"kinds_pick__{h}_*.npy")):
            arm = path.stem.split("__")[1]
            if arm in (f"{h}_rank1", f"{h}_full"):
                continue
            a = np.load(path)
            rows[arm] = {
                "pick": float(a.mean()), "minus_rank1": paired_bootstrap(a - top),
                "share_of_the_loss_with_all_constants": float((top - a).mean() / (top - const).mean()),
                "lpips_to_full": float(load("lpips_full", arm).mean()),
            }
            r = rows[arm]
            print(f"{arm:20s} pick {r['pick']:.2f} | minus top direction {r['minus_rank1']['mean']:+.3f} [{r['minus_rank1']['lo']:+.3f}, {r['minus_rank1']['hi']:+.3f}] | LPIPS to full {r['lpips_to_full']:.3f}", flush=True)
    RESULT.write_text(json.dumps(results, indent=1))


def tokens(n: int) -> None:
    """Per adapter and for the key and the value layers of cross-attention: the size of the coefficient
    at the start token, at the caption's words, at the end token and padding, and its plain mean over
    all 77 tokens, which is what one constant per layer stands for."""
    from transformers import CLIPTextModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer", revision=BASE_REV)
    text = CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder", revision=BASE_REV, use_safetensors=True).eval()
    ids = tok([it["caption"] for it in coco_items(n)], padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
    with torch.no_grad():
        x = text(ids)[0]  # captions x 77 x 768
    end = [int((row == tok.eos_token_id).nonzero()[0]) for row in ids]  # index of the end token
    results = json.loads(RESULT.read_text()) if RESULT.exists() else {}
    results["tokens"] = out = {"captions": n, "mean_tokens_with_start_and_end": float(np.mean(end)) + 1, "adapters": {}}
    for h in HOSTS:
        f = load_factors(h)
        const = {name: float(np.mean(v)) for name, v in torch.load(config.RUNS / "study" / f"constants_{h}.pt").items()}
        out["adapters"][h] = {}
        for label, suffix in (("cross_k", "attn2.to_k"), ("cross_v", "attn2.to_v")):
            names = [name for name in f if name.endswith(suffix)]
            rows = []
            for name in names:
                c = x @ f[name][2][0]  # captions x 77
                rows.append([
                    float(c[:, 0].mean()), float(np.mean([float(c[i, 1 : e].mean()) for i, e in enumerate(end)])),
                    float(np.mean([float(c[i, e:].mean()) for i, e in enumerate(end)])), float(c.mean()), const[name], float(c[:, 0].std()),
                ])
            r = np.array(rows)
            w = np.array([float(f[name][1][0] ** 2) for name in names])
            w /= w.sum()
            size = (np.abs(r[:, :5]) * w[:, None]).sum(0)
            out["adapters"][h][label] = g = {
                "layers": len(names),
                **{k: float(v) for k, v in zip(("start_token", "words", "end_and_padding", "all_77_tokens", "fitted_constant"), size)},
                "layers_where_start_and_mean_share_a_sign": float((np.sign(r[:, 0]) == np.sign(r[:, 3])).mean()),
                "largest_spread_of_start_token_over_captions": float(r[:, 5].max()),
            }
            print(f"{h:7s} {label} | start {g['start_token']:6.2f} | words {g['words']:5.2f} | end and padding {g['end_and_padding']:5.2f} | all 77 {g['all_77_tokens']:5.2f} | fitted constant {g['fitted_constant']:5.2f} | same sign in {100 * g['layers_where_start_and_mean_share_a_sign']:.0f}% of layers", flush=True)
    RESULT.write_text(json.dumps(results, indent=1))


def attention() -> None:
    """Per adapter: the share of cross-attention weight on the start token, on the prompt's words, and
    on the end token with the padding, averaged over the 16 cross-attention layers, their heads, the
    image positions, the steps and the development prompts."""
    from transformers import CLIPTokenizer

    b = bench()
    tok = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer", revision=BASE_REV)
    ids = tok(b.prompts, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
    end = [int((row == tok.eos_token_id).nonzero()[0]) for row in ids]  # index of the end token
    mods = {n: m for n, m in b.unet.named_modules() if n.endswith("attn2")}
    state, seen, shares = {"prompt": 0}, {}, []

    def weights(n):
        def hook(module, args, output):
            q, k = (seen[n, part].unflatten(-1, (module.heads, -1)).transpose(1, 2) for part in ("q", "k"))  # 1 x heads x positions x width
            p = (q @ k.transpose(-1, -2) * module.scale).softmax(-1).mean((0, 1, 2))  # 77
            e = end[state["prompt"]]
            shares.append([float(p[0]), float(p[1:e].sum()), float(p[e:].sum())])
        return hook

    results = json.loads(RESULT.read_text()) if RESULT.exists() else {}
    results["attention"] = out = {"mean_words_per_prompt": float(np.mean(end)) - 1}
    for h in HOSTS:
        shares.clear()
        b.install(rank1(load_factors(h)))
        handles = [m.register_forward_hook(weights(n)) for n, m in mods.items()]
        handles += [getattr(m, f"to_{part}").register_forward_hook(lambda module, args, output, key=(n, part): seen.__setitem__(key, output)) for n, m in mods.items() for part in ("q", "k")]
        try:
            for _ in b.trajectory(ADAPTERS[h].sampler, STEPS, before=lambda i, step: state.update(prompt=i)):
                pass
        finally:
            for handle in handles:
                handle.remove()
            b.install(None)
        out[h] = dict(zip(("start_token", "words", "end_and_padding"), map(float, np.mean(shares, 0))))
        RESULT.write_text(json.dumps(results, indent=1))
        print(f"{h:7s} | start token {out[h]['start_token']:.3f} | words {out[h]['words']:.3f} | end and padding {out[h]['end_and_padding']:.3f}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "100"
    if mode == "table":
        table()
    elif mode == "tokens":
        tokens(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
    else:
        run = {"measure": measure, "attention": attention}.get(mode) or (lambda: images(int(mode)))
        if best_device() == "cpu":  # a small job beside the GPU; nothing to queue for
            run()
        else:
            with device_lock():
                run()
