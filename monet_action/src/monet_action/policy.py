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
        # TOOL_RESULT: previously executed action(s) (+ validity flags) -> tokens before reasoning
        self.tool_result_projector = None
        tr = cfg.tool_result
        if tr.num_prev_actions > 0:
            self.tool_result_projector = RobotStateProjector(
                tr.num_prev_actions * (cfg.action_head.action_dim + 1), D, tr.num_tokens, tr.hidden_dim, 0.0,
                init_rms=embedding_rms(self.reasoner.embed_tokens))
        self.conditioner = LatentToActionConditioner(D, cfg.conditioner)
        self.action_head = build_action_head(cfg.action_head, cfg.state.state_dim, cfg.conditioner.cond_dim,
                                             use_state=cfg.state.state_to_action_head)
        self.prompt_builder = PromptBuilder(processor, cfg.prompt, cfg.monet.latent_convention, cfg.monet.num_latents,
                                            cfg.tool_result)
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
    def tool_result_features(self, prev_actions: Optional[torch.Tensor], prev_valid: Optional[torch.Tensor],
                             batch_size: int) -> Optional[torch.Tensor]:
        """[B, n, 7] normalised previous actions + [B, n] validity -> [B, n*8] projector input.
        During training a whole sample's TOOL_RESULT action is hidden with prob ``tool_result.dropout``."""
        if self.tool_result_projector is None:
            return None
        n, A, dev = self.cfg.tool_result.num_prev_actions, self.cfg.action_head.action_dim, self.device
        if prev_actions is None or prev_actions.shape[1] == 0:
            prev_actions = torch.zeros(batch_size, n, A, device=dev)
            prev_valid = torch.zeros(batch_size, n, device=dev)
        prev_actions, prev_valid = prev_actions.to(dev).float(), prev_valid.to(dev).float()
        if self.training and self.cfg.tool_result.dropout > 0:
            keep = (torch.rand(batch_size, 1, device=dev) >= self.cfg.tool_result.dropout).float()
            prev_valid = prev_valid * keep
        prev_actions = prev_actions * prev_valid.unsqueeze(-1)
        return torch.cat([prev_actions.flatten(1), prev_valid], dim=1)

    def encode(self, monet_inputs, state_norm: torch.Tensor, intervention: Optional[str] = None,
               generator: Optional[torch.Generator] = None, bank_keys=None, timer: Optional[Timer] = None,
               prev_actions: Optional[torch.Tensor] = None, prev_valid: Optional[torch.Tensor] = None):
        timer = timer or Timer(self.device)
        B = state_norm.shape[0]
        state_tokens = self.state_projector(state_norm) if self.state_projector is not None else None
        feats = self.tool_result_features(prev_actions, prev_valid, B)
        tool_tokens = self.tool_result_projector(feats) if feats is not None else None
        self._last_prompt_tokens = (state_tokens, tool_tokens)
        out = self.reasoner(monet_inputs, state_tokens, tool_tokens=tool_tokens, timer=timer)
        latents, latent_mask, vl_mask = out.latents, None, out.vl_mask
        if intervention in ("mask_vl", "mask_all"):  # latent-bypass diagnostics
            vl_mask = torch.zeros_like(vl_mask)
            if intervention == "mask_all" and latents.shape[1] > 0:
                latent_mask = torch.zeros(B, latents.shape[1], dtype=torch.bool, device=latents.device)
        elif intervention:
            latents, latent_mask = apply_latent_intervention(latents, None, intervention, generator=generator,
                                                             bank=self.latent_bank, bank_keys=bank_keys)
        with timer("conditioner"):
            cond = self.conditioner(out.vl_tokens, vl_mask, latents, latent_mask)
        return cond, out

    def _set_adapters(self, enabled: bool) -> None:
        from peft.tuners.tuners_utils import BaseTunerLayer

        for m in self.reasoner.model.modules():
            if isinstance(m, BaseTunerLayer):
                m.enable_adapters(enabled)

    def latent_retention_loss(self, inputs, latents: torch.Tensor) -> torch.Tensor:
        """Relative squared distance between LoRA latents and frozen-Monet latents (same inputs)."""
        state_tokens, tool_tokens = self._last_prompt_tokens
        with torch.no_grad():
            self._set_adapters(False)
            try:
                ref = self.reasoner(inputs, None if state_tokens is None else state_tokens.detach(),
                                    tool_tokens=None if tool_tokens is None else tool_tokens.detach()).latents
            finally:
                self._set_adapters(True)
        ref = ref.float()
        return ((latents.float() - ref) ** 2).sum(-1).mean() / (ref ** 2).sum(-1).mean().clamp_min(1e-6)

    # ------------------------------------------------------------------ training
    def compute_loss(self, batch: Dict, generator: Optional[torch.Generator] = None, t=None, noise=None):
        dev = self.device
        inputs = batch["monet_inputs"].to(dev)
        state = batch["state"].to(dev)
        with self._autocast():
            cond, out = self.encode(inputs, state, prev_actions=batch.get("prev_actions"),
                                    prev_valid=batch.get("prev_valid"))
            loss, metrics = self.action_head.loss(
                batch["actions"].to(dev), batch["loss_mask"].to(dev), cond.tokens, cond.mask,
                state=state if self.cfg.state.state_to_action_head else None, generator=generator, t=t, noise=noise)
            w = self.cfg.train.latent_retention_weight
            if self.training and w > 0 and out.latents.shape[1] > 0:
                ret = self.latent_retention_loss(inputs, out.latents)
                loss = loss + w * ret
                metrics["retention_loss"] = ret.detach()
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
            prev, valid = self.prev_actions_from_observations(observations)
        with self._autocast():
            cond, out = self.encode(inputs, state, intervention, generator, bank_keys, timer,
                                    prev_actions=prev, prev_valid=valid)
            with timer("action_head"):
                norm = self.action_head.sample(cond.tokens.float(), cond.mask,
                                               state=state if self.cfg.state.state_to_action_head else None,
                                               num_steps=num_steps, generator=generator, noise=noise)
        env = self.normalizer.to_env_action(norm)
        return PolicyPrediction(norm.float(), env, timer.result(), out.latents.detach())

    def prev_actions_from_observations(self, observations: List):
        """Right-aligned (newest last) normalised executed actions + validity flags from each
        observation's TOOL_RESULT history (``PolicyObservation.prev_actions``)."""
        n = self.cfg.tool_result.num_prev_actions
        if n == 0:
            return None, None
        A = self.cfg.action_head.action_dim
        prev = np.zeros((len(observations), n, A), np.float32)
        valid = np.zeros((len(observations), n), np.float32)
        for i, o in enumerate(observations):
            h = getattr(o, "prev_actions", None)
            if h is None or len(h) == 0:
                continue
            h = np.asarray(h, np.float32)[-n:]
            prev[i, n - len(h):] = self.normalizer.action.normalize(h)
            valid[i, n - len(h):] = 1.0
        return torch.from_numpy(prev), torch.from_numpy(valid)

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
        groups = {"monet_frozen": 0, "monet_lora": 0, "state_projector": 0, "tool_result_projector": 0,
                  "conditioner": 0, "action_head": 0, "learned_queries": 0}
        for n, p in self.named_parameters():
            if n.startswith("reasoner.model."):
                groups["monet_lora" if "lora_" in n else "monet_frozen"] += p.numel()
            elif n.startswith("reasoner.learned_queries"):
                groups["learned_queries"] += p.numel()
            elif n.startswith("state_projector"):
                groups["state_projector"] += p.numel()
            elif n.startswith("tool_result_projector"):
                groups["tool_result_projector"] += p.numel()
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
    for proj in (policy.state_projector, policy.tool_result_projector):
        if proj is not None:
            proj.to(device=device, dtype=head_dtype)
    if policy.reasoner.learned_queries is not None:
        policy.reasoner.learned_queries.data = policy.reasoner.learned_queries.data.to(device)
    return policy, info
