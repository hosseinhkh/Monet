"""GPU selection and (optional) up-front memory reservation.

* GPU pinning happens in scripts/common.sh: ``GPU_ID`` (default 0) sets ``CUDA_VISIBLE_DEVICES`` and
  ``MUJOCO_EGL_DEVICE_ID`` (simulator rendering) with ``CUDA_DEVICE_ORDER=PCI_BUS_ID``, so "GPU 0" is the
  same card that ``nvidia-smi`` calls 0. Inside Python the selected card is then always ``cuda:0``.
* ``MONET_GPU_RESERVE=<fraction>`` (e.g. 0.95) makes training / evaluation / audit processes claim that
  fraction of the card's memory right after the model is loaded: a large block is allocated and released
  into PyTorch's caching allocator, so it stays held by this process (visible in nvidia-smi) and is
  reused for activations. A margin (default 3 GB, ``MONET_GPU_RESERVE_MARGIN_GB``) stays free for CUDA
  kernels' small allocations and EGL rendering. Best effort: on a genuine out-of-memory PyTorch releases
  cached blocks and retries, so the reservation can shrink but never causes a crash by itself.
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


def reserve_gpu_memory(fraction: float, margin_gb: float = 3.0) -> float:
    """Hold ``fraction`` of the card's memory in this process's PyTorch cache. Returns GB newly reserved."""
    if fraction <= 0 or not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info(0)
    want = min(fraction * total - torch.cuda.memory_reserved(0), free - margin_gb * GiB)
    if want < GiB:
        return 0.0
    block = torch.empty(int(want), dtype=torch.uint8, device="cuda:0")
    del block  # returns to the caching allocator: still reserved by this process
    return want / GiB


def setup_gpu(device: str) -> Dict:
    """Call after the model is on the GPU. Applies MONET_GPU_RESERVE and logs what this process holds."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {"cuda": False}
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    frac = float(os.environ.get("MONET_GPU_RESERVE", "0") or 0)
    got = reserve_gpu_memory(frac, float(os.environ.get("MONET_GPU_RESERVE_MARGIN_GB", "3")))
    s = gpu_summary()
    s["reserve_fraction"], s["reserved_now_gb"] = frac, got
    log.info("GPU %s (%s, %.0f GB, visible=%s): this process holds %.1f GB, %.1f GB free%s", s["name"],
             s["capability"], s["total_gb"], s["CUDA_VISIBLE_DEVICES"], s["reserved_by_this_process_gb"], s["free_gb"],
             f" (reserved {got:.1f} GB up front, MONET_GPU_RESERVE={frac})" if frac > 0 else "")
    return s
