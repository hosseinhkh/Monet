"""Seeding, environment capture and small IO helpers."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def rng_state() -> Dict[str, Any]:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: Dict[str, Any]) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def _cmd(args) -> str:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "n/a"


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "not installed"


def environment_report() -> Dict[str, Any]:
    repo = Path(__file__).resolve().parents[3]
    rep = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "nvidia_driver": _cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "repo_commit": _cmd(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "repo_dirty": _cmd(["git", "-C", str(repo), "status", "--porcelain"]) not in ("", "n/a"),
    }
    for p in ("transformers", "peft", "accelerate", "safetensors", "numpy", "mujoco", "robosuite", "bddl",
              "pyarrow", "huggingface_hub", "libero"):
        rep[p] = _pkg_version(p)
    return rep


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def append_jsonl(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=_default) + "\n")


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
