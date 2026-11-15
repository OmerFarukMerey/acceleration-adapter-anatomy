"""Study 6: is a borrowed direction wrong, or only at the wrong strength?

For each host adapter, the direction the *other* methods agree on is installed at the host's
per-layer strength times one global factor. If some factor works, the direction transfers
and only a scale has to be found. A random direction at the same factors is the control.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pilots"))
from adapters import ADAPTERS, bench, load_factors  # noqa: E402
from c_lora_anatomy import line, score, sheet  # noqa: E402
from s2_function import HOSTS, consensus, random_direction, top  # noqa: E402

from wspd import config  # noqa: E402
from wspd.metrics import Evaluator  # noqa: E402
from wspd.store import device_lock  # noqa: E402

FACTORS = (0.35, 0.5, 0.7, 1.0, 1.4)
TARGETS = ["lcm", "tcd", "pcm", "hyper4"]
# `s6_strength.py <host> <factor> ...` adds more factors for one host to the saved results.
EXTRA = (sys.argv[1], tuple(float(a) for a in sys.argv[2:])) if len(sys.argv) > 2 else None


def scaled(parts, factor: float):
    return {n: (u, s * factor, vh) for n, (u, s, vh) in parts.items()}


def main() -> None:
    factors = {k: load_factors(k) for k in HOSTS}
    b = bench()
    evaluator = Evaluator(b.device)
    out = config.RESULTS / "study"
    path = out / "s6_strength.json"
    results = json.loads(path.read_text())["arms"] if EXTRA else {}
    for host in [EXTRA[0]] if EXTRA else TARGETS:
        f, sampler = factors[host], ADAPTERS[host].sampler
        shared = consensus([factors[k] for k in HOSTS if k != host], f)
        arms = {"none": None, "full": f, "rank_1": top(f)}
        arms.update({f"consensus_x{a}": scaled(shared, a) for a in (EXTRA[1] if EXTRA else FACTORS)})
        if not EXTRA:
            arms.update({f"random_x{a}": scaled(random_direction(f), a) for a in (0.5, 1.4)})
        images = {}
        for name, parts in arms.items():
            b.install(parts)
            images[name] = b.generate(sampler, ADAPTERS[host].steps)
        scored = score(evaluator, images, b.prompts)
        for name, r in scored.items():
            print(f"{host:7s} {line(name, r)}", flush=True)
        if EXTRA:
            results[host].update(scored)
        else:
            show = ["none", *[f"consensus_x{a}" for a in FACTORS], "rank_1", "full"]
            sheet(images, show, out / f"s6_strength_{host}.png")
            results[host] = {**scored, "sheet_columns": show}
        path.write_text(json.dumps({"arms": results}, indent=1))
    b.install(None)


if __name__ == "__main__":
    with device_lock():
        main()
