"""Latent-to-action conditioner: Monet features -> DiT cross-attention tokens [B, N, C].

Token layout (fixed order, logged in ``token_types``):
    [NULL] + [VL_0 .. VL_{L-1}] + [LAT_0 .. LAT_{K-1}]
* NULL is a learned always-valid token: every row has >= 1 valid key even if all latents are
  masked by an intervention and feature_source == latents_only.
* VL tokens: final-layer Monet hidden states of the (state-aware) prompt, projected by their
  own LayerNorm+Linear, plus a VL type embedding.
* LAT tokens: the K self-fed latent states in generation order, projected by a separate
  LayerNorm+Linear, plus a latent type embedding and a learned *order* embedding (k).
``feature_source`` selects which groups are visible; masked groups are excluded through the
attention mask (not zero-filled), so D0 (K=0) and D1 (K=10) share every parameter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

TOKEN_NULL, TOKEN_VL, TOKEN_LATENT = 0, 1, 2


@dataclass
class ConditioningTokens:
    tokens: torch.Tensor  # [B, N, C]
    mask: torch.Tensor  # [B, N] bool, True = valid key
    token_types: torch.Tensor  # [N] long (TOKEN_*)
    latent_order: torch.Tensor  # [N] long, k for latent tokens, -1 otherwise


class LatentToActionConditioner(nn.Module):
    def __init__(self, monet_dim: int, cfg):
        super().__init__()
        self.cfg = cfg
        C = cfg.cond_dim
        self.null_token = nn.Parameter(torch.zeros(1, 1, C))
        nn.init.normal_(self.null_token, std=0.02)
        self.vl_norm = nn.LayerNorm(monet_dim)
        self.vl_proj = nn.Linear(monet_dim, C)
        self.lat_norm = nn.LayerNorm(monet_dim)
        self.lat_proj = nn.Linear(monet_dim, C)
        self.type_embedding = nn.Embedding(3, C)
        self.order_embedding = nn.Embedding(cfg.max_latents, C)
        nn.init.normal_(self.type_embedding.weight, std=0.02)
        nn.init.normal_(self.order_embedding.weight, std=0.02)
        self.dropout = nn.Dropout(cfg.dropout)

    @property
    def use_vl(self) -> bool:
        return self.cfg.feature_source in ("vl_only", "vl_plus_latents")

    @property
    def use_latents(self) -> bool:
        return self.cfg.feature_source in ("latents_only", "vl_plus_latents")

    def forward(self, vl_tokens: Optional[torch.Tensor], vl_mask: Optional[torch.Tensor],
                latents: Optional[torch.Tensor], latent_mask: Optional[torch.Tensor] = None) -> ConditioningTokens:
        ref = latents if latents is not None else vl_tokens
        B, device = ref.shape[0], ref.device
        dtype = self.vl_proj.weight.dtype
        parts, masks, types, orders = [], [], [], []

        parts.append(self.null_token.expand(B, -1, -1).to(dtype) + self.type_embedding.weight[TOKEN_NULL])
        masks.append(torch.ones(B, 1, dtype=torch.bool, device=device))
        types.append(torch.full((1,), TOKEN_NULL, dtype=torch.long))
        orders.append(torch.full((1,), -1, dtype=torch.long))

        if self.use_vl and vl_tokens is not None and vl_tokens.shape[1] > 0:
            v = self.vl_proj(self.vl_norm(vl_tokens.to(dtype))) + self.type_embedding.weight[TOKEN_VL]
            parts.append(v)
            masks.append(vl_mask.to(torch.bool))
            types.append(torch.full((v.shape[1],), TOKEN_VL, dtype=torch.long))
            orders.append(torch.full((v.shape[1],), -1, dtype=torch.long))

        if self.use_latents and latents is not None and latents.shape[1] > 0:
            K = latents.shape[1]
            if K > self.cfg.max_latents:
                raise ValueError(f"K={K} exceeds max_latents={self.cfg.max_latents}")
            k_idx = torch.arange(K, device=device)
            z = self.lat_proj(self.lat_norm(latents.to(dtype)))
            z = z + self.type_embedding.weight[TOKEN_LATENT] + self.order_embedding(k_idx)[None]
            parts.append(z)
            lm = torch.ones(B, K, dtype=torch.bool, device=device) if latent_mask is None else latent_mask.to(torch.bool)
            if self.training and self.cfg.latent_block_dropout > 0:
                keep = torch.rand(B, 1, device=device) >= self.cfg.latent_block_dropout
                lm = lm & keep
            masks.append(lm)
            types.append(torch.full((K,), TOKEN_LATENT, dtype=torch.long))
            orders.append(torch.arange(K, dtype=torch.long))

        tokens = self.dropout(torch.cat(parts, 1))
        return ConditioningTokens(
            tokens=tokens, mask=torch.cat(masks, 1),
            token_types=torch.cat(types).to(device), latent_order=torch.cat(orders).to(device),
        )


# --------------------------------------------------------------------------------------
# D3 latent interventions (evaluation-time perturbation diagnostics; OOD by construction)
# --------------------------------------------------------------------------------------
class LatentBank:
    """FIFO bank of latents from *other* observations, used by the ``swap`` intervention."""

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.items = []  # (key, tensor[K, D])

    def add(self, latents: torch.Tensor, keys) -> None:
        for i, k in enumerate(keys):
            self.items.append((k, latents[i].detach().to("cpu", torch.float32)))
        self.items = self.items[-self.capacity:]

    def sample_excluding(self, key, generator: Optional[torch.Generator] = None) -> Optional[torch.Tensor]:
        cands = [t for k, t in self.items if k != key]
        if not cands:
            return None
        j = int(torch.randint(len(cands), (1,), generator=generator).item())
        return cands[j]


def apply_latent_intervention(latents: torch.Tensor, latent_mask: Optional[torch.Tensor], spec: Optional[str], *,
                              generator: Optional[torch.Generator] = None, bank: Optional[LatentBank] = None,
                              bank_keys=None):
    """Return (latents', latent_mask') for D3 diagnostics.

    zero          replace every latent by 0
    mask          drop latent tokens from the conditioning (attention mask)
    shuffle_order permute the K latent positions (order embeddings stay at their slots)
    truncate:k    keep only the first k latents (mask the rest)
    gaussian      replace with N(mean, std) matched per sample to the real latents
    swap          replace with latents from a *different* task/episode (needs ``bank``)
    """
    if spec is None or spec in ("", "none") or latents is None or latents.shape[1] == 0:
        return latents, latent_mask
    B, K, D = latents.shape
    mask = torch.ones(B, K, dtype=torch.bool, device=latents.device) if latent_mask is None else latent_mask.clone()
    if spec == "zero":
        return torch.zeros_like(latents), mask
    if spec == "mask":
        return latents, torch.zeros_like(mask)
    if spec == "shuffle_order":
        perm = torch.randperm(K, generator=generator).to(latents.device)
        if K > 1 and torch.equal(perm.cpu(), torch.arange(K)):
            perm = torch.roll(perm, 1)
        return latents[:, perm], mask
    if spec.startswith("truncate:"):
        k = int(spec.split(":", 1)[1])
        mask[:, k:] = False
        return latents, mask
    if spec == "gaussian":
        mu = latents.float().mean(dim=(1, 2), keepdim=True)
        sd = latents.float().std(dim=(1, 2), keepdim=True)
        noise = torch.randn(latents.shape, generator=generator).to(latents.device)
        return (mu + sd * noise).to(latents.dtype), mask
    if spec == "swap":
        if bank is None:
            raise ValueError("swap intervention needs a LatentBank")
        out = latents.clone()
        for i in range(B):
            other = bank.sample_excluding(bank_keys[i] if bank_keys is not None else None, generator)
            if other is not None and other.shape[0] == K:
                out[i] = other.to(latents.device, latents.dtype)
        return out, mask
    raise ValueError(f"unknown intervention {spec}")
