"""Study 2: do the directions agree in function, not only in weight space?

For each host adapter, at 4 steps with its own sampler:
  none / full / rank_1 / tail     the prerequisite: is the host one direction per layer?
  swap_<donor>                    the donor's top direction per layer, at the host's strength
  consensus                       the direction the *other* methods agree on, at the host's strength
  random                          a random direction per layer at the host's strength
Recovery is measured against the host's full adapter on the same seeds.
"""
from __future__ import annotations

import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pilots"))
from adapters import ADAPTERS, bench, factorize, load_factors  # noqa: E402
from c_lora_anatomy import line, score, sheet  # noqa: E402

from wspd import config  # noqa: E402
from wspd.metrics import Evaluator  # noqa: E402
from wspd.store import device_lock  # noqa: E402

HOSTS = ["lcm", "tcd", "pcm", "hyper4", "flash"]  # one per method; independent releases
SEEDS = int(os.environ.get("STUDY_SEEDS", "1"))  # seeds per prompt


def top(f):
    return {n: (u[:, :1], s[:1], vh[:1]) for n, (u, s, vh) in f.items()}


def tail(f):
    return {n: (u[:, 1:], s[1:], vh[1:]) for n, (u, s, vh) in f.items()}


def swap(donor, host):
    """The donor's top direction at the host's top strength, on the layers both have."""
    return {n: (donor[n][0][:, :1], host[n][1][:1], donor[n][2][:1]) for n in host if n in donor}


def consensus(others: list, host):
    """Top direction of the sum of the others' unit rank-1 changes, at the host's top strength."""
    out = {}
    for n, (_, s, _) in host.items():
        have = [f[n] for f in others if n in f]
        if not have:
            continue
        u, _, vh = factorize(torch.cat([f[0][:, :1] for f in have], 1), torch.cat([f[2][:1] for f in have], 0), 1.0)
        out[n] = (u[:, :1], s[:1], vh[:1])
    return out


def random_direction(host, seed: int = 0):
    out = {}
    for n, (u, s, vh) in host.items():
        gen = torch.Generator().manual_seed(zlib.crc32(n.encode()) + seed)
        a = torch.nn.functional.normalize(torch.randn(u.shape[0], 1, generator=gen), dim=0)
        b = torch.nn.functional.normalize(torch.randn(1, vh.shape[1], generator=gen), dim=1)
        out[n] = (a, s[:1], b)
    return out


def main() -> None:
    factors = {k: load_factors(k) for k in HOSTS}
    b = bench()
    evaluator = Evaluator(b.device)
    prompts = b.prompts * SEEDS
    results, sheets = {}, {}

    def run(parts, sampler, steps=4, guidance=1.0):
        b.install(parts)
        return np.concatenate([b.generate(sampler, steps, guidance, seed_offset=k) for k in range(SEEDS)])

    # References from the base model alone, for prompt alignment.
    refs = {"base_50_steps_guided": run(None, "ddim", 50, 7.5), "base_4_steps_guided": run(None, "ddim", 4, 7.5)}
    reference = {k: {"clip": float(evaluator.clip_similarity(v, prompts).mean())} for k, v in refs.items()}
    print(reference, flush=True)

    for host in HOSTS:
        f, sampler = factors[host], ADAPTERS[host].sampler
        others = [k for k in HOSTS if k != host]
        arms = {
            "none": lambda: None,
            "full": lambda: f,
            "rank_1": lambda: top(f),
            "tail": lambda: tail(f),
            "random": lambda: random_direction(f),
            "consensus": lambda: consensus([factors[k] for k in others], f),
            **{f"swap_{d}": (lambda d=d: swap(factors[d], f)) for d in others},
        }
        images, changed = {}, {}
        for name, build in arms.items():
            parts = build()
            changed[name] = len(parts or {})
            images[name] = run(parts, sampler)
            print(f"{host}: generated {name}", flush=True)
        results[host] = score(evaluator, images, prompts)
        for name, r in results[host].items():
            r["layers_changed"] = changed[name]
            print(f"{host:7s} {line(name, r)}", flush=True)
        sheets[host] = images
    b.install(None)

    out = config.RESULTS / "study"
    for host, images in sheets.items():
        show = ["none", "random", "tail", "rank_1", "consensus", *[f"swap_{d}" for d in HOSTS if d != host], "full"]
        sheet(images, show, out / f"s2_function_{host}.png")
        results[host]["sheet_columns"] = show
    (out / "s2_function.json").write_text(json.dumps({"steps": 4, "n_prompts": len(b.prompts), "seeds": SEEDS, "hosts": HOSTS, "reference": reference, "arms": results}, indent=1))


if __name__ == "__main__":
    with device_lock():
        main()
