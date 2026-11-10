"""Flow-matching primitives (GR00T N1.7 contract, re-implemented and unit-tested).

Convention (identical to ``Gr00tN1d7ActionHead`` at Isaac-GR00T 7aa0bca):
    x_t      = (1 - t) * eps + t * a        t=0: pure noise, t=1: data
    v_target = a - eps
    t_disc   = floor(t * num_buckets)        discrete timestep fed to the DiT
    training t ~ (1 - Beta(alpha, beta)) * s  (alpha=1.5, beta=1.0, s=0.999)
    inference: Euler from t=0 with dt = 1/N, t_i = i/N, i = 0..N-1
"""
from __future__ import annotations

from typing import Callable, Optional

import torch


def sample_time(
    batch_size: int,
    *,
    schedule: str = "beta",
    alpha: float = 1.5,
    beta: float = 1.0,
    s: float = 0.999,
    device=None,
    dtype=torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample flow time t in [0, s] (beta) or [0, 1) (uniform). Shape [B]."""
    if schedule == "uniform":
        t = torch.rand(batch_size, generator=generator)
    elif schedule == "beta":
        # Beta sampling via two Gammas so that an explicit generator can be used.
        ga = torch._standard_gamma(torch.full((batch_size,), float(alpha)), generator=generator)
        gb = torch._standard_gamma(torch.full((batch_size,), float(beta)), generator=generator)
        b = ga / (ga + gb)
        t = (1.0 - b) * s
    else:
        raise ValueError(f"unknown time schedule {schedule}")
    return t.to(device=device, dtype=dtype)


def interpolate(actions: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x_t = (1 - t) * eps + t * a with t broadcast over [B, H, A]."""
    t = t.view(-1, *([1] * (actions.dim() - 1))).to(actions.dtype)
    return (1.0 - t) * noise + t * actions


def velocity_target(actions: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return actions - noise


def discretize_time(t: torch.Tensor, num_buckets: int = 1000) -> torch.Tensor:
    return (t * num_buckets).long().clamp_(0, num_buckets - 1)


def masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], delta: float = 1e-6
) -> torch.Tensor:
    """sum(M * (pred - target)^2) / (sum(M) + delta). ``mask`` broadcastable to pred."""
    err = (pred.float() - target.float()) ** 2
    if mask is None:
        return err.mean()
    m = mask.to(err.dtype)
    if m.dim() == err.dim() - 1:
        m = m.unsqueeze(-1)
    m = m.expand_as(err)
    return (err * m).sum() / (m.sum() + delta)


@torch.no_grad()
def euler_integrate(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    shape,
    num_steps: int,
    *,
    num_buckets: int = 1000,
    device=None,
    dtype=torch.float32,
    generator: Optional[torch.Generator] = None,
    noise: Optional[torch.Tensor] = None,
    action_mask: Optional[torch.Tensor] = None,
    return_trajectory: bool = False,
):
    """Integrate dx/dt = v(x, t) from Gaussian noise at t=0 to t=1 with N Euler steps.

    ``velocity_fn(x, t_disc)`` gets the current sample and the discretised timestep [B].
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if noise is None:
        noise = torch.randn(shape, generator=generator, dtype=torch.float32)
    x = noise.to(device=device, dtype=dtype)
    if action_mask is not None:
        x = x * action_mask
    dt = 1.0 / num_steps
    traj = [x.clone()] if return_trajectory else None
    B = shape[0]
    for i in range(num_steps):
        t_cont = i / float(num_steps)
        t_disc = torch.full((B,), int(t_cont * num_buckets), device=x.device, dtype=torch.long)
        v = velocity_fn(x, t_disc)
        if action_mask is not None:
            v = v * action_mask
        x = x + dt * v.to(x.dtype)
        if traj is not None:
            traj.append(x.clone())
    if return_trajectory:
        return x, traj
    return x
