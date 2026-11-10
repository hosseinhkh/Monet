import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from monet_action.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg_path():
    return str(ROOT / "configs" / "tiny_cpu.yaml")


@pytest.fixture()
def cfg(cfg_path):
    return load_config(cfg_path)


@pytest.fixture(scope="session")
def tiny_model():
    from monet_action.monet_loader import build_tiny_monet

    return build_tiny_monet(seed=0)


@pytest.fixture(scope="session")
def processor():
    from monet_action.config import MonetConfig
    from monet_action.monet_loader import load_processor

    path = os.environ.get("MONET_TOKENIZER_PATH")
    try:
        return load_processor(MonetConfig(), path)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Monet tokenizer/processor files unavailable ({e}); set MONET_TOKENIZER_PATH")


@pytest.fixture(scope="session")
def data_root():
    p = Path(os.environ.get("MONET_ACTION_DATA", ROOT / "data" / "libero_spatial_image"))
    if not (p / "meta" / "info.json").exists():
        pytest.skip(f"LeRobot LIBERO dataset not found at {p} (scripts/download_data.sh)")
    return str(p)


def random_views(B, n_views=2, size=128, seed=0):
    rng = np.random.default_rng(seed)
    return [[rng.integers(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(n_views)] for _ in range(B)]


INSTR = ["pick up the black bowl next to the plate and place it on the plate", "pick up the bowl"]


def build_inputs(processor, cfg, conv="sft", K=10, B=2, seed=0):
    from monet_action.prompt import PromptBuilder

    cfg.monet.latent_convention = conv
    pb = PromptBuilder(processor, cfg.prompt, conv, K)
    return pb.build(random_views(B, seed=seed), INSTR[:B]), pb


def sim_available() -> bool:
    try:
        from monet_action.libero_env import _ensure_gl_backend

        _ensure_gl_backend()
        import libero.libero  # noqa: F401
        import robosuite  # noqa: F401

        return True
    except Exception:
        return False
