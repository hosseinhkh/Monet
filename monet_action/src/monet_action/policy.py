"""MonetActionPolicy: Monet latent reasoning -> conditioner -> CALL_ACTION -> flow-matching DiT.

    RGB (agent + wrist) + instruction + 8D state
        -> RobotStateProjector (state tokens placed *inside* the prompt, before reasoning)
        -> Monet customised Qwen2.5-VL prefill (state-aware VL context)
        -> K genuine self-fed latent steps                         [B, K, D]
        -> LatentToActionConditioner (VL + latents, typed+masked)  [B, N, C]
        -> DiT velocity field + state encoder, Euler integration   [B, H=8, A=7] (normalised)
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .conditioning import LatentBank, LatentToActionConditioner, apply_latent_intervention
from .dit_action import build_action_head
from .monet_backbone import MonetLatentReasoner
from .normalization import ActionStateNormalizer
from .profiling import Timer
from .prompt import PromptBuilder
from .state_context import RobotStateProjector, embedding_rms


@dataclass
class PolicyPrediction:
    norm_actions: torch.Tensor  # [B, H, A] normalised
    env_actions: np.ndarray  # [B, H, A] simulator commands
    timings_ms: Dict[str, float] = field(default_factory=dict)
    latents: Optional[torch.Tensor] = None  # [B, K, D] (for D3 banks / diagnostics)


def lora_target_regex(cfg) -> str:
    layers = "|".join(str(i) for i in cfg.lora_layers)
    mods = "|".join(cfg.lora_target_modules)
    return rf".*language_model\.layers\.({layers})\.self_attn\.({mods})"


class MonetActionPolicy(nn.Module):
    def __init__(self, cfg, monet_model: nn.Module, processor, normalizer: Optional[ActionStateNormalizer] = None):
        super().__init__()
        self.cfg = cfg
        # ---- freeze Monet, then (optionally) inject LoRA into selected decoder layers
        for p in monet_model.parameters():
            p.requires_grad_(False)
        if cfg.monet.lora_enabled:
            from peft import LoraConfig, inject_adapter_in_model

            lcfg = LoraConfig(r=cfg.monet.lora_r, lora_alpha=cfg.monet.lora_alpha, lora_dropout=cfg.monet.lora_dropout,
                              target_modules=lora_target_regex(cfg.monet), bias="none")
            monet_model = inject_adapter_in_model(lcfg, monet_model)
            n = sum(1 for k, _ in monet_model.named_parameters() if "lora_" in k)
            if n == 0:
                raise RuntimeError("LoRA injection matched no modules")
            for k, p in monet_model.named_parameters():
                p.requires_grad_("lora_" in k)
                if "lora_" in k:
                    p.data = p.data.float()  # fp32 adapter weights (PEFT casts activations back)
        if cfg.monet.gradient_checkpointing:
            monet_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.reasoner = MonetLatentReasoner(monet_model, cfg.monet, cfg.conditioner)
        D = self.reasoner.hidden_size
        self.state_projector = None
        if cfg.state.inject_into_monet:
            self.state_projector = RobotStateProjector(
                cfg.state.state_dim, D, cfg.prompt.num_state_tokens, cfg.state.hidden_dim, cfg.state.dropout,
                init_rms=embedding_rms(self.reasoner.embed_tokens))
        self.conditioner = LatentToActionConditioner(D, cfg.conditioner)
        self.action_head = build_action_head(cfg.action_head, cfg.state.state_dim, cfg.conditioner.cond_dim,
                                             use_state=cfg.state.state_to_action_head)
        self.prompt_builder = PromptBuilder(processor, cfg.prompt, cfg.monet.latent_convention, cfg.monet.num_latents)
        self.normalizer = normalizer
        self.latent_bank = LatentBank()

    # ------------------------------------------------------------------ modes
    def train(self, mode: bool = True):
        super().train(mode)
        m = self.cfg.monet
        monet_train = mode and (m.lora_enabled or m.gradient_checkpointing)
        self.reasoner.model.train(monet_train)
        return self

    @property
    def device(self):
        return next(self.action_head.parameters()).device

    def _autocast(self):
        if self.device.type == "cuda" and self.cfg.train.mixed_precision == "bf16":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # ------------------------------------------------------------------ encode
    def encode(self, monet_inputs, state_norm: torch.Tensor, intervention: Optional[str] = None,
               generator: Optional[torch.Generator] = None, bank_keys=None, timer: Optional[Timer] = None):
        timer = timer or Timer(self.device)
        state_tokens = None
        if self.state_projector is not None:
            state_tokens = self.state_projector(state_norm)
        with timer("monet"):
            out = self.reasoner(monet_inputs, state_tokens)
        latents, latent_mask = out.latents, None
        if intervention:
            latents, latent_mask = apply_latent_intervention(latents, None, intervention, generator=generator,
                                                             bank=self.latent_bank, bank_keys=bank_keys)
        with timer("conditioner"):
            cond = self.conditioner(out.vl_tokens, out.vl_mask, latents, latent_mask)
        return cond, out

    # ------------------------------------------------------------------ training
    def compute_loss(self, batch: Dict, generator: Optional[torch.Generator] = None, t=None, noise=None):
        dev = self.device
        inputs = batch["monet_inputs"].to(dev)
        state = batch["state"].to(dev)
        with self._autocast():
            cond, out = self.encode(inputs, state)
            loss, metrics = self.action_head.loss(
                batch["actions"].to(dev), batch["loss_mask"].to(dev), cond.tokens, cond.mask,
                state=state if self.cfg.state.state_to_action_head else None, generator=generator, t=t, noise=noise)
        metrics = {k: float(v) for k, v in metrics.items()}
        metrics["loss"] = float(loss.detach())
        metrics["n_cond_tokens"] = int(cond.tokens.shape[1])
        metrics["latent_norm"] = float(out.latents.float().norm(dim=-1).mean()) if out.latents.shape[1] else 0.0
        return loss, metrics

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, observations: List, num_steps: Optional[int] = None, generator: Optional[torch.Generator] = None,
                intervention: Optional[str] = None, bank_keys=None, noise: Optional[torch.Tensor] = None) -> PolicyPrediction:
        if self.normalizer is None:
            raise RuntimeError("policy has no normalizer (load it from the checkpoint)")
        dev = self.device
        timer = Timer(dev)
        with timer("preprocess"):
            views = [[o.images[k] for k in self.cfg.prompt.camera_keys] for o in observations]
            inputs = self.prompt_builder.build(views, [o.instruction for o in observations]).to(dev)
            state = self.normalizer.state.normalize(
                torch.from_numpy(np.stack([o.state for o in observations]))).float().to(dev)
        with self._autocast():
            cond, out = self.encode(inputs, state, intervention, generator, bank_keys, timer)
            with timer("action_head"):
                norm = self.action_head.sample(cond.tokens.float(), cond.mask,
                                               state=state if self.cfg.state.state_to_action_head else None,
                                               num_steps=num_steps, generator=generator, noise=noise)
        env = self.normalizer.to_env_action(norm)
        return PolicyPrediction(norm.float(), env, timer.result(), out.latents.detach())

    # ------------------------------------------------------------------ checkpoints
    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}

    def load_trainable_state_dict(self, sd: Dict[str, torch.Tensor], strict: bool = True) -> None:
        own = dict(self.named_parameters())
        missing = [n for n, p in own.items() if p.requires_grad and n not in sd]
        unexpected = [k for k in sd if k not in own]
        if strict and (missing or unexpected):
            raise RuntimeError(f"trainable state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        with torch.no_grad():
            for k, v in sd.items():
                if k in own:
                    own[k].copy_(v.to(own[k].dtype))

    def parameter_report(self) -> Dict[str, int]:
        groups = {"monet_frozen": 0, "monet_lora": 0, "state_projector": 0, "conditioner": 0, "action_head": 0,
                  "learned_queries": 0}
        for n, p in self.named_parameters():
            if n.startswith("reasoner.model."):
                groups["monet_lora" if "lora_" in n else "monet_frozen"] += p.numel()
            elif n.startswith("reasoner.learned_queries"):
                groups["learned_queries"] += p.numel()
            elif n.startswith("state_projector"):
                groups["state_projector"] += p.numel()
            elif n.startswith("conditioner"):
                groups["conditioner"] += p.numel()
            elif n.startswith("action_head"):
                groups["action_head"] += p.numel()
        groups["trainable_total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return groups


def build_policy(cfg, device="cpu", normalizer=None, processor=None, tokenizer_path=None):
    from .monet_loader import load_monet, load_processor

    model, info = load_monet(cfg.monet, device=device)
    if processor is None:
        processor = load_processor(cfg.monet, tokenizer_path)
    if device != "cpu" and next(model.parameters()).device.type != torch.device(device).type:
        model = model.to(device)
    policy = MonetActionPolicy(cfg, model, processor, normalizer)
    head_dtype = torch.float32
    for mod in (policy.action_head, policy.conditioner):
        mod.to(device=device, dtype=head_dtype)
    if policy.state_projector is not None:
        policy.state_projector.to(device=device, dtype=head_dtype)
    if policy.reasoner.learned_queries is not None:
        policy.reasoner.learned_queries.data = policy.reasoner.learned_queries.data.to(device)
    return policy, info
