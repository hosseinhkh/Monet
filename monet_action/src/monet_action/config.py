"""Typed experiment configuration with YAML loading and ``_base_`` inheritance.

Every field that changes the scientific meaning of a run (K, feature source,
recurrence convention, gradient policy, LoRA targets, DiT size, flow-matching
schedule, data split, evaluation protocol) lives here so that it is saved next
to every checkpoint and every result file.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# --------------------------------------------------------------------------------------
# Sub-configs
# --------------------------------------------------------------------------------------
@dataclass
class MonetConfig:
    # Path or HF id of the Monet checkpoint. "tiny-random" builds a tiny randomly
    # initialised model that runs Monet's *actual* modified Qwen2.5-VL code (tests/CPU).
    model_path: str = "NOVAglow646/Monet-7B"
    model_revision: Optional[str] = "39c1e320a3e15c357bff911fda263923edbdda37"
    # Where the authors' customised modeling file lives (vendored in this repo).
    modeling_file: Optional[str] = None  # default: <repo>/monet_qwen_model/modeling_qwen2_5_vl_monet.py
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    # ---- latent reasoning ----
    num_latents: int = 10  # K. 0 == no latent reasoning (D0)
    # "sft": <abs_vis_token> keeps its token embedding, K <abs_vis_token_pad> slots are fed
    #        the previous position's final (post-norm) hidden state. Matches the authors'
    #        differentiable HF forward (Qwen2_5_VLModel.forward(latent_mode=True)).
    # "vllm": the <abs_vis_token> slot itself is overwritten with the previous hidden state
    #        and K-1 further slots follow. Matches the authors' vLLM inference runner.
    latent_convention: str = "sft"
    # "cached": prefill + K single-token steps on a KV cache (fast; no HF grad checkpointing)
    # "recompute": K+1 full forwards on the growing sequence (works with grad checkpointing)
    recurrence_impl: str = "cached"
    # "bptt": no detach anywhere in the feedback loop (full backprop through K steps)
    # "truncated": detach the fed-back vector (what the authors' SFT latent_mode does)
    feedback_grad: str = "bptt"
    # Which hidden layer to read latent / VL features from. -1 == final post-norm hidden
    # state, which is exactly the vector Monet feeds back.
    latent_layer: int = -1
    vl_layer: int = -1
    # "self_fed": genuine Monet recurrence. "learned_query": D4 matched-compute control
    # (K trainable query embeddings at the same slots, no hidden-state feedback).
    reasoning_mode: str = "self_fed"
    learned_query_parallel: bool = True  # D4: process K query slots in one forward
    # ---- what is trainable ----
    freeze_vision: bool = True
    gradient_checkpointing: bool = False  # only valid with recurrence_impl="recompute"
    lora_enabled: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_layers: List[int] = field(default_factory=lambda: list(range(20, 28)))
    # ---- tiny-random model (tests / CPU) ----
    tiny_hidden_size: int = 64
    tiny_num_layers: int = 2


@dataclass
class PromptConfig:
    system_prompt: str = "You are a helpful assistant."
    template: str = (
        "Robot state: {state}\nTask: {instruction}\n"
        "Think in latent space about how to complete the task, then call the action tool."
    )
    # Placeholder token whose embeddings are replaced by projected proprioception.
    state_placeholder_token: str = "<|fim_pad|>"
    num_state_tokens: int = 2
    camera_keys: List[str] = field(default_factory=lambda: ["image", "wrist_image"])
    image_size: int = 224  # each view is resized to image_size x image_size (multiple of 28)
    append_latent_end_token: bool = False


@dataclass
class StateConfig:
    state_dim: int = 8
    hidden_dim: int = 1024
    # Pass the state through Monet (before reasoning) via the projector.
    inject_into_monet: bool = True
    # Also give the state directly to the DiT state encoder.
    state_to_action_head: bool = True
    dropout: float = 0.0


@dataclass
class ConditionerConfig:
    feature_source: str = "vl_plus_latents"  # latents_only | vl_only | vl_plus_latents
    cond_dim: int = 512  # C
    max_latents: int = 32  # size of the latent-order embedding table
    vl_max_tokens: int = 512  # keep the last N valid prompt tokens
    vl_token_select: str = "all"  # all | image_only | text_only
    dropout: float = 0.0
    # Training-time latent dropout (whole latent block masked with this prob). 0 == off.
    latent_block_dropout: float = 0.0


@dataclass
class ActionHeadConfig:
    type: str = "dit"  # dit | mlp (D5 engineering control)
    action_dim: int = 7
    action_horizon: int = 8
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    interleave_self_attention: bool = True
    num_timestep_buckets: int = 1000
    # flow matching
    time_schedule: str = "beta"  # beta (GR00T: t=(1-Beta(a,b))*s) | uniform
    beta_alpha: float = 1.5
    beta_beta: float = 1.0
    noise_s: float = 0.999
    num_inference_steps: int = 4
    mlp_hidden: int = 1024


@dataclass
class DataConfig:
    source: str = "lerobot"
    root: str = "data/libero_spatial_image"  # local LeRobot v3 dataset dir
    repo_id: str = "lerobot/libero_spatial_image"
    repo_revision: Optional[str] = "d86c0b94922572b3b657e1d1a3d01f0952ddeb46"
    val_fraction: float = 0.1
    split_seed: int = 0
    task_indices: Optional[List[int]] = None  # None == all tasks
    max_episodes_per_task: Optional[int] = None
    file_limit: Optional[int] = None  # only read the first N parquet files (smoke/CPU)
    action_norm: str = "q99"  # q99 | minmax | meanstd
    state_norm: str = "meanstd"
    augment: bool = False
    num_workers: int = 4
    # overfit/smoke: restrict to a fixed set of windows
    max_windows: Optional[int] = None


@dataclass
class TrainConfig:
    output_dir: str = "runs/debug"
    seed: int = 0
    max_steps: int = 20000
    batch_size: int = 8
    grad_accum: int = 2
    lr_head: float = 1e-4
    lr_lora: float = 5e-5
    lr_state: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    mixed_precision: str = "bf16"  # bf16 | no
    log_every: int = 20
    val_every: int = 1000
    val_batches: int = 50
    save_every: int = 2000
    keep_last: int = 3
    keep_every: int = 10000  # checkpoints at multiples of this step are never deleted
    init_from: Optional[str] = None  # warm-start trainable weights (e.g. D2 from D1 stage A)
    gradient_audit_steps: int = 1  # assert finite/nonzero grads on the first N steps
    # G1 overfit evidence: every N steps, fixed-noise open-loop metrics on the *training*
    # windows (deterministic, so the curve is comparable step to step). 0 == off.
    overfit_eval_every: int = 0
    deterministic: bool = False
    compile: bool = False


@dataclass
class EvalConfig:
    suite: str = "libero_spatial"
    task_ids: Optional[List[int]] = None  # None == all 10
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    episodes_per_task: int = 10
    # "seed_offset": seed i uses benchmark init states [i*E, (i+1)*E) -> 3 seeds x 10 eps = 30
    # distinct, variant-matched init states per task. "fixed": every seed uses states [0, E)
    # (then seeds only change the DiT sampling noise).
    init_state_policy: str = "seed_offset"
    max_steps: int = 220  # standard libero_spatial cap
    num_wait_steps: int = 10  # settle steps with a no-op action after reset (not counted)
    execution_prefix: int = 1  # E
    num_inference_steps: Optional[int] = None  # override flow steps (2/4/8)
    num_parallel_envs: int = 10
    camera_size: int = 256
    save_videos: int = 2  # per task per seed
    intervention: Optional[str] = None  # D3: zero|mask|shuffle_order|truncate:k|gaussian|swap
    output_dir: str = "results/eval"
    gpu_hourly_cost_usd: Optional[float] = None


@dataclass
class ExperimentConfig:
    name: str = "debug"
    variant: str = "D1"
    description: str = ""
    monet: MonetConfig = field(default_factory=MonetConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    state: StateConfig = field(default_factory=StateConfig)
    conditioner: ConditionerConfig = field(default_factory=ConditionerConfig)
    action_head: ActionHeadConfig = field(default_factory=ActionHeadConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ---------------------------------------------------------------- helpers
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def validate(self) -> None:
        m, c = self.monet, self.conditioner
        assert m.latent_convention in ("sft", "vllm"), m.latent_convention
        assert m.recurrence_impl in ("cached", "recompute"), m.recurrence_impl
        assert m.feedback_grad in ("bptt", "truncated"), m.feedback_grad
        assert m.reasoning_mode in ("self_fed", "learned_query"), m.reasoning_mode
        assert c.feature_source in ("latents_only", "vl_only", "vl_plus_latents"), c.feature_source
        if c.feature_source == "latents_only" and m.num_latents == 0:
            raise ValueError("feature_source=latents_only requires num_latents > 0")
        if m.gradient_checkpointing and m.recurrence_impl == "cached":
            raise ValueError(
                "HF gradient checkpointing disables the KV cache in training mode; "
                "use monet.recurrence_impl=recompute together with gradient_checkpointing"
            )
        if m.num_latents > c.max_latents:
            raise ValueError("conditioner.max_latents must be >= monet.num_latents")
        if self.prompt.image_size % 28 != 0:
            raise ValueError("prompt.image_size must be a multiple of 28 (Qwen2.5-VL patch*merge)")
        assert self.action_head.type in ("dit", "mlp")
        if not self.state.inject_into_monet and not self.state.state_to_action_head:
            raise ValueError("state must reach the policy through at least one path")


# --------------------------------------------------------------------------------------
# YAML loading
# --------------------------------------------------------------------------------------
def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_with_base(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    bases = raw.pop("_base_", None)
    if bases is None:
        return raw
    if isinstance(bases, str):
        bases = [bases]
    merged: Dict[str, Any] = {}
    for b in bases:
        merged = _deep_update(merged, _load_yaml_with_base((path.parent / b).resolve()))
    return _deep_update(merged, raw)


def _from_dict(cls, data: Dict[str, Any]):
    kwargs = {}
    names = {f.name: f for f in fields(cls)}
    for k, v in (data or {}).items():
        if k not in names:
            raise KeyError(f"Unknown config key '{k}' for {cls.__name__}")
        ftype = names[k].type
        sub = _SUBCONFIGS.get(k) if cls is ExperimentConfig else None
        if sub is not None and isinstance(v, dict):
            kwargs[k] = _from_dict(sub, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


_SUBCONFIGS = {
    "monet": MonetConfig,
    "prompt": PromptConfig,
    "state": StateConfig,
    "conditioner": ConditionerConfig,
    "action_head": ActionHeadConfig,
    "data": DataConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}


def parse_overrides(overrides: List[str]) -> Dict[str, Any]:
    """``["monet.num_latents=0", "train.max_steps=10"]`` -> nested dict (YAML-typed values)."""
    out: Dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item}")
        key, val = item.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return out


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> ExperimentConfig:
    data: Dict[str, Any] = {}
    if path is not None:
        data = _load_yaml_with_base(Path(path).resolve())
    if overrides:
        data = _deep_update(data, parse_overrides(overrides))
    cfg = _from_dict(ExperimentConfig, data)
    cfg.validate()
    return cfg


def config_from_dict(data: Dict[str, Any]) -> ExperimentConfig:
    cfg = _from_dict(ExperimentConfig, data)
    cfg.validate()
    return cfg
