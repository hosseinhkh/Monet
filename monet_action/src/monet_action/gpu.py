"""GPU selection and (optional) up-front memory reservation.

* GPU pinning happens in scripts/common.sh: ``GPU_ID`` (default 0) sets ``CUDA_VISIBLE_DEVICES`` and
  ``MUJOCO_EGL_DEVICE_ID`` (simulator rendering) with ``CUDA_DEVICE_ORDER=PCI_BUS_ID``, so "GPU 0" is the
  same card that ``nvidia-smi`` calls 0. Inside Python the selected card is then always ``cuda:0``.
* ``MONET_GPU_RESERVE=<fraction>`` (e.g. 0.5) makes training / evaluation / audit processes claim that
  fraction of the card's memory as their first GPU action (before data and weights load): a large block
  is allocated and released into PyTorch's caching allocator, so it stays held by this process (visible
  in nvidia-smi) and is reused for the weights and activations. A margin (default 3 GB, ``MONET_GPU_RESERVE_MARGIN_GB``) stays free for CUDA
  kernels' small allocations and EGL rendering. Best effort: on a genuine out-of-memory PyTorch releases
  cached blocks and retries, so the reservation can shrink but never causes a crash by itself.
* The same fraction is also a hard per-process cap (``torch.cuda.set_per_process_memory_fraction``) so a
  run never grows into memory that other users on the card rely on. ``MONET_GPU_LIMIT`` overrides the cap
  (e.g. reserve 0.4 but allow up to 0.5); ``MONET_GPU_LIMIT=0`` disables it.
* Machine-local settings (GPU_ID, MONET_GPU_RESERVE, HF_HOME, ...) can live in ``monet_action/env.local``.
* ``MONET_GPU_HOLD=1``: a background holder (:mod:`monet_action.gpu_hold`) keeps the share claimed between
  runs; :func:`setup_gpu` first takes a lease so the holder hands its memory over to this process.
"""
from __future__ import annotations

import logging
import os
from typing import Dict

import torch

GiB = 1024 ** 3
log = logging.getLogger("monet_action.gpu")


def gpu_summary() -> Dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    return {"cuda": True, "name": p.name, "capability": f"sm_{p.major}{p.minor}", "total_gb": total / GiB,
            "free_gb": free / GiB, "visible_devices": torch.cuda.device_count(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "MUJOCO_EGL_DEVICE_ID": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            "reserved_by_this_process_gb": torch.cuda.memory_reserved(0) / GiB}


def limit_fraction() -> float:
    """Per-process cap: MONET_GPU_LIMIT if set, else the MONET_GPU_RESERVE fraction (0 = no cap)."""
    lim = os.environ.get("MONET_GPU_LIMIT")
    if lim not in (None, ""):
        return float(lim)
    return float(os.environ.get("MONET_GPU_RESERVE", "0") or 0)


def apply_memory_limit(fraction: float) -> float:
    """Cap this process's PyTorch allocations at ``fraction`` of the card (good neighbour on a shared GPU):
    going over raises a normal CUDA out-of-memory error in *this* process instead of starving others."""
    if fraction <= 0 or fraction >= 1 or not torch.cuda.is_available():
        return 0.0
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return fraction


def reserve_gpu_memory(fraction: float, margin_gb: float = 3.0, capped: bool = False) -> float:
    """Hold ``fraction`` of the card's memory in this process's PyTorch cache. Returns GB newly reserved.

    ``capped``: a per-process limit at the same fraction is active, so stay 1 GB below it (the allocator
    rounds block sizes up and needs room for small allocations).
    """
    if fraction <= 0 or not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info(0)
    target = fraction * total - (GiB if capped else 0)
    want = min(target - torch.cuda.memory_reserved(0), free - margin_gb * GiB)
    while want >= GiB:
        try:
            block = torch.empty(int(want), dtype=torch.uint8, device="cuda:0")
            del block  # returns to the caching allocator: still reserved by this process
            return want / GiB
        except torch.cuda.OutOfMemoryError:
            want -= GiB  # someone else allocated meanwhile; take a little less
    return 0.0


def setup_gpu(device: str) -> Dict:
    """First GPU action of a process (before data, weights or CUDA work): takes the holder's lease
    (MONET_GPU_HOLD), then applies MONET_GPU_LIMIT / MONET_GPU_RESERVE and logs the result. Idempotent."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {"cuda": False}
    from .gpu_hold import acquire_lease

    leased = acquire_lease()  # no-op without a running holder
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    frac = float(os.environ.get("MONET_GPU_RESERVE", "0") or 0)
    cap = apply_memory_limit(limit_fraction())
    total = torch.cuda.get_device_properties(0).total_memory
    got = reserve_gpu_memory(frac, float(os.environ.get("MONET_GPU_RESERVE_MARGIN_GB", "3")), capped=bool(cap))
    s = gpu_summary()
    s["reserve_fraction"], s["reserved_now_gb"], s["limit_fraction"] = frac, got, cap
    s["limit_gb"] = cap * total / GiB if cap else None
    s["holder_lease"] = leased
    log.info("GPU %s (%s, %.0f GB, visible=%s): this process holds %.1f GB, %.1f GB free%s%s%s", s["name"],
             s["capability"], s["total_gb"], s["CUDA_VISIBLE_DEVICES"], s["reserved_by_this_process_gb"], s["free_gb"],
             f" | reserved {got:.1f} GB up front (MONET_GPU_RESERVE={frac})" if frac > 0 else "",
             f" | capped at {cap:.0%} = {cap * total / GiB:.0f} GB" if cap else "",
             " | share handed over by the GPU holder" if leased else "")
    if frac > 0 and s["reserved_by_this_process_gb"] < 0.8 * (frac * total / GiB - 1):
        log.warning("only %.0f of %.0f GB reserved: other processes use the card (nvidia-smi); training may OOM",
                    s["reserved_by_this_process_gb"], frac * total / GiB)
    return s
