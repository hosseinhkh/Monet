"""Latency / memory measurement helpers (CUDA-synchronised when on GPU)."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, List

import numpy as np
import torch


def _sync(device) -> None:
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class Timer:
    """Accumulating named wall-clock timer; ``result()`` returns milliseconds."""

    def __init__(self, device=None):
        self.device = device
        self.ms: Dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        _sync(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.device)
            self.ms[name] = self.ms.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0

    def result(self) -> Dict[str, float]:
        return dict(self.ms)


def percentiles(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"p50": float("nan"), "p95": float("nan"), "mean": float("nan"), "n": 0}
    a = np.asarray(xs, dtype=np.float64)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "mean": float(a.mean()), "n": int(a.size)}


def reset_peak_memory(device=None) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device=None) -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / 1024 ** 3
    return float("nan")
