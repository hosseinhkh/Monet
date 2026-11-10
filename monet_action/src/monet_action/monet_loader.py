"""Load Monet-7B through the authors' customised Qwen2.5-VL modeling file.

We never use stock ``transformers`` Qwen2.5-VL classes for Monet: the authors' file
(``monet_qwen_model/modeling_qwen2_5_vl_monet.py`` in this repository, pinned to
NOVAglow646/Monet@08939998) is imported under a private module name, so the stock module in
``sys.modules`` is left untouched (unlike the authors' ``apply_qwen2_5_monet.py`` monkey patch).

``build_tiny_monet`` builds a randomly initialised model with the *same* class, the same
special-token ids and the same vision patching as Monet-7B but a few-MB text/vision stack. It
exercises the real Monet code paths on CPU (tests, smoke runs). It is never a substitute for
the real checkpoint in any reported result.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MONET_MODULE_NAME = "monet_action_vendored.modeling_qwen2_5_vl_monet"

# Special tokens of Monet-7B (tokenizer added_tokens.json @ 39c1e32; verified in G0).
LATENT_PAD_ID = 151665  # <abs_vis_token_pad>
LATENT_START_ID = 151666  # <abs_vis_token>
LATENT_END_ID = 151667  # </abs_vis_token>
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_ID = 151652
VISION_END_ID = 151653
PAD_TOKEN_ID = 151643
IM_START_ID = 151644
ASSISTANT_ID = 77091
MONET_VOCAB_SIZE = 151670


def default_modeling_file() -> Path:
    env = os.environ.get("MONET_MODELING_FILE")
    if env:
        return Path(env)
    return REPO_ROOT / "monet_qwen_model" / "modeling_qwen2_5_vl_monet.py"


def load_monet_module(path: Optional[str] = None):
    if MONET_MODULE_NAME in sys.modules:
        return sys.modules[MONET_MODULE_NAME]
    path = Path(path) if path else default_modeling_file()
    if not path.exists():
        raise FileNotFoundError(f"Monet modeling file not found: {path}")
    spec = importlib.util.spec_from_file_location(MONET_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MONET_MODULE_NAME] = module
    # the authors' file triggers many harmless transformers auto-docstring prints at import
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def tiny_monet_config(hidden_size: int = 64, num_layers: int = 2):
    from transformers import Qwen2_5_VLConfig

    heads = 4
    head_dim = hidden_size // heads
    half = head_dim // 2
    mrope = [half // 4, (half - half // 4) // 2, half - half // 4 - (half - half // 4) // 2]
    text = dict(
        vocab_size=MONET_VOCAB_SIZE, hidden_size=hidden_size, intermediate_size=2 * hidden_size,
        num_hidden_layers=num_layers, num_attention_heads=heads, num_key_value_heads=2,
        max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=1_000_000.0,
        rope_scaling={"type": "mrope", "mrope_section": mrope}, tie_word_embeddings=False,
        use_sliding_window=False, sliding_window=None, max_window_layers=num_layers,
    )
    vision = dict(
        depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=hidden_size,
        patch_size=14, spatial_merge_size=2, temporal_patch_size=2, in_channels=3, window_size=112,
        fullatt_block_indexes=[1], tokens_per_second=2,
    )
    cfg = Qwen2_5_VLConfig(
        text_config=text, vision_config=vision, image_token_id=IMAGE_TOKEN_ID, video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_ID, vision_end_token_id=VISION_END_ID,
        **text,
    )
    # Monet-specific config entries (same values as Monet-7B config.json)
    cfg.latent_token_id = LATENT_PAD_ID
    cfg.latent_start_id = LATENT_START_ID
    cfg.latent_end_id = LATENT_END_ID
    cfg.answer_start_pattern = [IM_START_ID, ASSISTANT_ID]
    cfg.pad_token_id = PAD_TOKEN_ID
    return cfg


def build_tiny_monet(hidden_size: int = 64, num_layers: int = 2, seed: int = 0, modeling_file: Optional[str] = None,
                     attn_implementation: str = "sdpa"):
    mod = load_monet_module(modeling_file)
    cfg = tiny_monet_config(hidden_size, num_layers)
    cfg._attn_implementation = attn_implementation
    g = torch.random.get_rng_state()
    torch.manual_seed(seed)
    model = mod.Qwen2_5_VLForConditionalGeneration(cfg)
    torch.random.set_rng_state(g)
    model.eval()
    return model


def load_monet(cfg, device: Optional[str] = None) -> Tuple[torch.nn.Module, dict]:
    """Return (model, load_info). ``cfg`` is a :class:`MonetConfig`. ``device`` loads weights directly
    onto that device (avoids a 16 GB CPU staging copy for Monet-7B)."""
    if cfg.model_path == "tiny-random":
        model = build_tiny_monet(cfg.tiny_hidden_size, cfg.tiny_num_layers, modeling_file=cfg.modeling_file,
                                 attn_implementation="sdpa" if cfg.attn_implementation == "flash_attention_2" else cfg.attn_implementation)
        return model, {"tiny_random": True, "missing_keys": [], "unexpected_keys": []}
    mod = load_monet_module(cfg.modeling_file)
    model, info = mod.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_path, revision=cfg.model_revision, torch_dtype=_torch_dtype(cfg.dtype),
        attn_implementation=cfg.attn_implementation, output_loading_info=True,
        device_map=device if device and str(device).startswith("cuda") else None,
    )
    missing = [k for k in info.get("missing_keys", [])]
    unexpected = [k for k in info.get("unexpected_keys", [])]
    if missing:
        raise RuntimeError(f"Monet checkpoint is missing {len(missing)} weights, e.g. {missing[:5]}")
    for name in ("latent_token_id", "latent_start_id", "latent_end_id", "answer_start_pattern"):
        if not hasattr(model.config, name):
            raise RuntimeError(f"Monet config lacks '{name}' - is this really a Monet checkpoint?")
    if (model.config.latent_token_id, model.config.latent_start_id, model.config.latent_end_id) != (
            LATENT_PAD_ID, LATENT_START_ID, LATENT_END_ID):
        raise RuntimeError("Unexpected Monet latent special-token ids")
    model.eval()
    return model, {"tiny_random": False, "missing_keys": missing, "unexpected_keys": unexpected}


def load_processor(cfg, tokenizer_path: Optional[str] = None):
    """Qwen2.5-VL processor (tokenizer + image processor) shipped with the Monet checkpoint."""
    from transformers import AutoProcessor

    path = tokenizer_path or os.environ.get("MONET_TOKENIZER_PATH")
    revision = None
    if path is None:
        path = cfg.model_path if cfg.model_path != "tiny-random" else "NOVAglow646/Monet-7B"
        revision = cfg.model_revision
    proc = AutoProcessor.from_pretrained(path, revision=revision, use_fast=True)
    tok = proc.tokenizer
    for t, i in (("<abs_vis_token_pad>", LATENT_PAD_ID), ("<abs_vis_token>", LATENT_START_ID),
                 ("</abs_vis_token>", LATENT_END_ID), ("<|image_pad|>", IMAGE_TOKEN_ID)):
        if tok.convert_tokens_to_ids(t) != i:
            raise RuntimeError(f"tokenizer id mismatch for {t}")
    return proc
