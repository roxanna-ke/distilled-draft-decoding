"""Wall-clock timing with optional CUDA sync so GPU work is not under-counted."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class Timer:
    """Cumulative timer. Use `with timer:` to add an interval to total_s."""

    total_s: float = 0.0
    n: int = 0
    _t0: float = 0.0

    def __enter__(self) -> "Timer":
        cuda_sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        cuda_sync()
        self.total_s += time.perf_counter() - self._t0
        self.n += 1


@contextmanager
def stopwatch():
    """Yields a callable that returns elapsed seconds since entry (CUDA-synced)."""
    cuda_sync()
    t0 = time.perf_counter()

    def elapsed() -> float:
        cuda_sync()
        return time.perf_counter() - t0

    yield elapsed
