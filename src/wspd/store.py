"""The device lock that keeps two scripts from using the model at the same time."""
from __future__ import annotations

import fcntl
from contextlib import contextmanager

from . import config


@contextmanager
def device_lock():
    """Serialise model use so that parallel scripts queue instead of exhausting memory."""
    config.RUNS.mkdir(parents=True, exist_ok=True)
    with open(config.RUNS / ".device.lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
