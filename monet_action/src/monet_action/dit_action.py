"""Compact GR00T-N1.7-style flow-matching DiT action expert (pure PyTorch re-implementation).

Design reference: NVIDIA Isaac-GR00T @ 7aa0bca50f1097f8b17f5c274c98264b72a1419f
  gr00t/model/gr00t_n1d7/gr00t_n1d7.py   (Gr00tN1d7ActionHead: forward / get_action_with_features)
  gr00t/model/modules/dit.py              (DiT, BasicTransformerBlock, AdaLayerNorm, TimestepEncoder)
  gr00t/model/modules/embodiment_conditioned_mlp.py (MultiEmbodimentActionEncoder, CategorySpecificMLP)
Apache-2.0. No GR00T weights are loaded: the conditioning width, action/state dims, horizon
and depth all differ from the released checkpoint, so the module is trained from scratch.

What is kept from GR00T: state token + noisy-action tokens as the DiT sequence; action
encoder W1/W2/W3 with a sinusoidal encoding of the *discretised* flow time; learned position
embedding on action tokens; AdaLayerNorm(t) pre-norm blocks; interleaved cross-attention
(even blocks, to the conditioning tokens) and self-attention (odd blocks); output AdaLN
shift/scale + projection; MLP action decoder reading the last H tokens; velocity prediction.
What is changed: single embodiment (CategorySpecific* -> nn.Linear); diffusers is not a
dependency; cross-attention always uses an explicit key-padding mask (GR00T's
``strict_vl_padding_mask``) and the conditioner guarantees >= 1 valid key per row;
no 132-D padding, no state dropout by default.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_matching import discretize_time, euler_integrate, interpolate, masked_mse, sample_time, velocity_target


# --------------------------------------------------------------------------------------
# Time encodings
# --------------------------------------------------------------------------------------
def diffusers_timestep_embedding(timesteps: torch.Tensor, dim: int = 256, max_period: float = 10000.0,
                                 flip_sin_to_cos: bool = True, downscale_freq_shift: float = 1.0) -> torch.Tensor:
    """Same formula as diffusers.get_timestep_embedding (used by GR00T's TimestepEncoder)."""
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half - downscale_freq_shift)
    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def sinusoidal_encoding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """GR00T SinusoidalPositionalEncoding: t [B, T] -> [B, T, dim] = [sin, cos]."""
    half = dim // 2
    exponent = -torch.arange(half, dtype=torch.float32, device=t.device) * (math.log(10000.0) / half)
    freqs = t.float().unsqueeze(-1) * exponent.exp()
    return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class TimestepEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(256, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, t_disc: torch.Tensor) -> torch.Tensor:
        e = diffusers_timestep_embedding(t_disc, 256).to(self.linear_1.weight.dtype)
        return self.linear_2(F.silu(self.linear_1(e)))


# --------------------------------------------------------------------------------------
# Encoders / decoder
# --------------------------------------------------------------------------------------
class ActionEncoder(nn.Module):
    """GR00T MultiEmbodimentActionEncoder for one embodiment."""

    def __init__(self, action_dim: int, dim: int):
        super().__init__()
        self.dim = dim
        self.W1 = nn.Linear(action_dim, dim)
        self.W2 = nn.Linear(2 * dim, dim)
        self.W3 = nn.Linear(dim, dim)

    def forward(self, actions: torch.Tensor, t_disc: torch.Tensor) -> torch.Tensor:
        B, T, _ = actions.shape
        a = self.W1(actions)
        tau = sinusoidal_encoding(t_disc.view(B, 1).expand(B, T), self.dim).to(a.dtype)
        x = self.W2(torch.cat([a, tau], dim=-1))
        x = x * torch.sigmoid(x)  # swish
        return self.W3(x)


class MLP2(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden)
        self.layer2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


# --------------------------------------------------------------------------------------
# DiT blocks
# --------------------------------------------------------------------------------------
class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.linear = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(F.silu(temb)).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kv_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.dh = heads, dim // heads
        kv_dim = kv_dim or dim
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(kv_dim, dim)
        self.to_v = nn.Linear(kv_dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x, context=None, key_mask: Optional[torch.Tensor] = None):
        B, T, _ = x.shape
        ctx = x if context is None else context
        S = ctx.shape[1]
        q = self.to_q(x).view(B, T, self.heads, self.dh).transpose(1, 2)
        k = self.to_k(ctx).view(B, S, self.heads, self.dh).transpose(1, 2)
        v = self.to_v(ctx).view(B, S, self.heads, self.dh).transpose(1, 2)
        attn_mask = None
        if key_mask is not None:
            attn_mask = key_mask[:, None, None, :].to(torch.bool)  # True = attend
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.to_out(out.transpose(1, 2).reshape(B, T, -1))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, cross_dim: Optional[int], ffn_mult: int, dropout: float):
        super().__init__()
        self.is_cross = cross_dim is not None
        self.norm1 = AdaLayerNorm(dim)
        self.attn = Attention(dim, heads, kv_dim=cross_dim)
        self.drop = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim), nn.GELU(approximate="tanh"), nn.Dropout(dropout),
            nn.Linear(ffn_mult * dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x, temb, context=None, context_mask=None, self_mask=None):
        h = self.norm1(x, temb)
        if self.is_cross:
            h = self.attn(h, context=context, key_mask=context_mask)
        else:
            h = self.attn(h, key_mask=self_mask)
        x = x + self.drop(h)
        x = x + self.ff(self.norm3(x))
        return x


def _mask_tokens(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    return x if mask is None else x * mask.unsqueeze(-1).to(x.dtype)


class FlowMatchingDiT(nn.Module):
    """Velocity field v_theta(x_t, t, cond_tokens, state) -> [B, H, A]."""

    def __init__(self, action_dim=7, state_dim=8, horizon=8, dim=512, num_layers=6, heads=8,
                 cond_dim=512, ffn_mult=4, dropout=0.1, interleave_self_attention=True,
                 num_timestep_buckets=1000, use_state=True, max_seq_len=64):
        super().__init__()
        self.action_dim, self.state_dim, self.horizon = action_dim, state_dim, horizon
        self.dim, self.use_state = dim, use_state
        self.num_timestep_buckets = num_timestep_buckets
        self.state_encoder = MLP2(state_dim, dim, dim) if use_state else None
        self.action_encoder = ActionEncoder(action_dim, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        self.timestep_encoder = TimestepEncoder(dim)
        blocks = []
        for i in range(num_layers):
            self_attn = interleave_self_attention and (i % 2 == 1)
            blocks.append(DiTBlock(dim, heads, None if self_attn else cond_dim, ffn_mult, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.norm_out = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(dim, 2 * dim)
        self.proj_out_2 = nn.Linear(dim, dim)
        self.action_decoder = MLP2(dim, dim, action_dim)

    def forward(self, noisy_actions, t_disc, cond, cond_mask, state=None, action_token_mask=None):
        B, H, _ = noisy_actions.shape
        a = self.action_encoder(noisy_actions, t_disc)
        a = a + self.position_embedding(torch.arange(H, device=a.device))[None]
        tokens = [a]
        hidden_mask = action_token_mask
        if self.use_state:
            if state is None:
                raise ValueError("DiT was built with use_state=True but no state was given")
            s = self.state_encoder(state.unsqueeze(1).to(a.dtype))
            tokens.insert(0, s)
            if hidden_mask is not None:
                hidden_mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=a.device), hidden_mask.bool()], 1)
        x = torch.cat(tokens, dim=1)
        x = _mask_tokens(x, hidden_mask)
        temb = self.timestep_encoder(t_disc)
        cond = cond.to(x.dtype)
        for blk in self.blocks:
            if blk.is_cross:
                x = blk(x, temb, context=cond, context_mask=cond_mask)
            else:
                x = blk(x, temb, self_mask=hidden_mask)
            x = _mask_tokens(x, hidden_mask)
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + scale[:, None]) + shift[:, None]
        x = self.proj_out_2(_mask_tokens(x, hidden_mask))
        v = self.action_decoder(x[:, -H:])
        if action_token_mask is not None:
            v = v * action_token_mask.unsqueeze(-1).to(v.dtype)
        return v


class DiTActionHead(nn.Module):
    """Flow-matching training loss + Euler sampling around :class:`FlowMatchingDiT`."""

    kind = "dit"

    def __init__(self, cfg, state_dim: int, cond_dim: int, use_state: bool):
        super().__init__()
        self.cfg = cfg
        self.model = FlowMatchingDiT(
            action_dim=cfg.action_dim, state_dim=state_dim, horizon=cfg.action_horizon,
            dim=cfg.hidden_size, num_layers=cfg.num_layers, heads=cfg.num_heads, cond_dim=cond_dim,
            ffn_mult=cfg.ffn_mult, dropout=cfg.dropout, interleave_self_attention=cfg.interleave_self_attention,
            num_timestep_buckets=cfg.num_timestep_buckets, use_state=use_state,
        )

    def loss(self, actions, loss_mask, cond, cond_mask, state=None, generator=None, t=None, noise=None):
        """Masked flow-matching loss. ``actions`` are normalised, [B, H, A]."""
        B = actions.shape[0]
        if t is None:
            t = sample_time(B, schedule=self.cfg.time_schedule, alpha=self.cfg.beta_alpha,
                            beta=self.cfg.beta_beta, s=self.cfg.noise_s, generator=generator)
        t = t.to(device=actions.device, dtype=actions.dtype)
        if noise is None:
            noise = torch.randn(actions.shape, generator=generator, dtype=torch.float32)
        noise = noise.to(device=actions.device, dtype=actions.dtype)
        x_t = interpolate(actions, noise, t)
        v_star = velocity_target(actions, noise)
        t_disc = discretize_time(t, self.cfg.num_timestep_buckets)
        v = self.model(x_t, t_disc, cond, cond_mask, state=state)
        loss = masked_mse(v, v_star, loss_mask)
        return loss, {"fm_loss": loss.detach(), "t_mean": t.mean().detach()}

    @torch.no_grad()
    def sample(self, cond, cond_mask, state=None, num_steps=None, generator=None, noise=None):
        num_steps = num_steps or self.cfg.num_inference_steps
        B = cond.shape[0]
        shape = (B, self.cfg.action_horizon, self.cfg.action_dim)

        def vfn(x, t_disc):
            return self.model(x, t_disc, cond, cond_mask, state=state)

        return euler_integrate(vfn, shape, num_steps, num_buckets=self.cfg.num_timestep_buckets,
                               device=cond.device, dtype=torch.float32, generator=generator, noise=noise)


class MLPActionHead(nn.Module):
    """D5 engineering control: pooled conditioning + state -> chunk, plain masked MSE regression.

    Never a substitute for the DiT main result.
    """

    kind = "mlp"

    def __init__(self, cfg, state_dim: int, cond_dim: int, use_state: bool):
        super().__init__()
        self.cfg, self.use_state = cfg, use_state
        in_dim = cond_dim + (cfg.hidden_size if use_state else 0)
        self.state_encoder = MLP2(state_dim, cfg.hidden_size, cfg.hidden_size) if use_state else None
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.mlp_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden), nn.GELU(),
            nn.Linear(cfg.mlp_hidden, cfg.action_horizon * cfg.action_dim),
        )

    def _predict(self, cond, cond_mask, state):
        m = cond_mask.unsqueeze(-1).to(cond.dtype)
        pooled = (cond * m).sum(1) / m.sum(1).clamp_min(1.0)
        feats = [pooled]
        if self.use_state:
            feats.append(self.state_encoder(state.to(cond.dtype)))
        out = self.net(torch.cat(feats, -1))
        return out.view(-1, self.cfg.action_horizon, self.cfg.action_dim)

    def loss(self, actions, loss_mask, cond, cond_mask, state=None, generator=None, **_):
        pred = self._predict(cond, cond_mask, state)
        loss = masked_mse(pred, actions, loss_mask)
        return loss, {"mse_loss": loss.detach()}

    @torch.no_grad()
    def sample(self, cond, cond_mask, state=None, **_):
        return self._predict(cond, cond_mask, state).float()


def build_action_head(cfg, state_dim: int, cond_dim: int, use_state: bool) -> nn.Module:
    if cfg.type == "dit":
        return DiTActionHead(cfg, state_dim, cond_dim, use_state)
    if cfg.type == "mlp":
        return MLPActionHead(cfg, state_dim, cond_dim, use_state)
    raise ValueError(cfg.type)
