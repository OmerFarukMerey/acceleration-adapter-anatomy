"""Study 5: could agreement come from a shared starting point instead of a shared solution?

LoRA training starts from a random input-side matrix. If two adapters were trained from the
same starting matrix, or one was fine-tuned from the other, their weak directions (which
training barely moves) would still span the same input subspace. This measures the overlap
of the input subspaces spanned by directions 9 and beyond, against the value for unrelated
subspaces. No image generation.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from adapters import ACCEL, load_factors  # noqa: E402

from wspd import config  # noqa: E402

SKIP = 8


def main() -> None:
    factors = {k: load_factors(k) for k in ACCEL}
    result = {}
    for a, b in combinations(ACCEL, 2):
        tail, expected, out_tail = [], [], []
        for n in factors[a]:
            if n not in factors[b]:
                continue
            (ua, _, va), (ub, _, vb) = factors[a][n], factors[b][n]
            va, vb, ua, ub = va[SKIP:], vb[SKIP:], ua[:, SKIP:], ub[:, SKIP:]
            k = min(len(va), len(vb))
            tail.append(float((va @ vb.T).pow(2).sum()) / k)
            out_tail.append(float((ua.T @ ub).pow(2).sum()) / k)
            expected.append(max(len(va), len(vb)) / va.shape[1])
        result[f"{a}|{b}"] = r = {"input_tail_overlap": float(np.mean(tail)), "output_tail_overlap": float(np.mean(out_tail)), "unrelated": float(np.mean(expected))}
        print(f"{a:12s} {b:12s} input tail overlap {r['input_tail_overlap']:.3f} | output tail overlap {r['output_tail_overlap']:.3f} | unrelated {r['unrelated']:.3f}", flush=True)
    (config.RESULTS / "study" / "s5_lineage.json").write_text(json.dumps({"skip_top": SKIP, "pairs": result}, indent=1))


if __name__ == "__main__":
    main()
