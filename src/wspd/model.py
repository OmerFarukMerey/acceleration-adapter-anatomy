"""Device choice. Importing this module turns autograd off for the whole process."""
from __future__ import annotations

import torch

# The scripts generate images without wrapping each call in no_grad. Those that need
# gradients (s10, s14, s15) switch autograd back on around the lines that use it.
torch.set_grad_enabled(False)


def best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"
