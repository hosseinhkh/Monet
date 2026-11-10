"""Process-level defaults shared by scripts/common.sh and the ``python -m monet_action.<tool>`` CLIs.

Must run before torch initialises CUDA, so CLI modules call :func:`apply_defaults` before importing torch.
Existing environment variables always win.

* ``GPU_ID`` (default ``0``, or ``auto`` = the card with the most free memory, numbered as in nvidia-smi)
  -> ``CUDA_VISIBLE_DEVICES`` + ``MUJOCO_EGL_DEVICE_ID`` with ``CUDA_DEVICE_ORDER=PCI_BUS_ID``.
* ``MUJOCO_GL`` / ``PYOPENGL_PLATFORM``: ``egl`` when an NVIDIA GPU is present, else ``osmesa``.
* ``monet_action/env.local`` (optional, git-ignored) is read first, e.g. ``MONET_GPU_RESERVE=0.5``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ENV_LOCAL = Path(__file__).resolve().parents[2] / "env.local"


def read_env_local(path: Path = ENV_LOCAL) -> Dict[str, str]:
    """Machine-local settings (git-ignored): ``KEY=VALUE`` or ``export KEY=VALUE`` lines, ``#`` comments."""
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out


def nvidia_smi_gpus() -> List[Tuple[int, str, float, float]]:
    """[(index, name, free_GB, total_GB)] in nvidia-smi (PCI bus) order; [] without nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.free,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 4:
            gpus.append((int(p[0]), p[1], float(p[2]) / 1024, float(p[3]) / 1024))
    return gpus


def pick_gpu(spec: Optional[str]) -> Optional[str]:
    spec = (spec or "0").strip()
    if spec != "auto":
        return spec
    gpus = nvidia_smi_gpus()
    return str(max(gpus, key=lambda g: g[2])[0]) if gpus else "0"


def apply_defaults() -> None:
    for k, v in read_env_local().items():  # env.local < explicit environment variables
        os.environ.setdefault(k, v)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = pick_gpu(os.environ.get("GPU_ID"))
    first = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0].strip()
    if first:
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", first)
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl" if shutil.which("nvidia-smi") else "osmesa"
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
