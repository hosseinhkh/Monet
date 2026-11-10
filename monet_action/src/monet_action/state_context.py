"""RobotStateProjector: normalised 8D proprioception -> n state tokens in Monet's embedding space.

The tokens replace the embeddings of ``num_state_tokens`` placeholder tokens inside the user
turn, i.e. *before* the latent reasoning starts, so every self-fed latent step can attend to
the robot state. The output is RMS-matched to Monet's token-embedding scale at init (a learned
gain initialised from the mean RMS of the input embedding table) so that the projected tokens
are in-distribution for the frozen backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RobotStateProjector(nn.Module):
    def __init__(self, state_dim: int, monet_dim: int, num_tokens: int, hidden_dim: int = 1024,
                 dropout: float = 0.0, init_rms: float = 1.0):
        super().__init__()
        self.state_dim, self.monet_dim, self.num_tokens = state_dim, monet_dim, num_tokens
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tokens * monet_dim),
        )
        self.token_pos = nn.Parameter(torch.zeros(num_tokens, monet_dim))
        self.gain = nn.Parameter(torch.tensor(float(init_rms)))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state [B, S] (normalised) -> [B, n, D_monet]."""
        B = state.shape[0]
        x = self.net(state.to(self.net[0].weight.dtype)).view(B, self.num_tokens, self.monet_dim)
        x = x + self.token_pos
        rms = x.float().pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt().to(x.dtype)
        return x * rms * self.gain


def embedding_rms(embedding: nn.Embedding, sample: int = 4096) -> float:
    w = embedding.weight
    idx = torch.linspace(0, w.shape[0] - 1, min(sample, w.shape[0])).long().to(w.device)
    return float(w[idx].float().pow(2).mean(-1).sqrt().mean().item())
