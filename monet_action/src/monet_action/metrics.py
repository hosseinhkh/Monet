"""Offline (open-loop) metrics, success-rate statistics and cost/latency aggregation."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

from .flow_matching import masked_mse
from .profiling import percentiles


# --------------------------------------------------------------------------- offline
@torch.no_grad()
def offline_metrics(policy, loader, max_batches: int = 50, num_steps: int = 4, seed: int = 1234) -> Dict[str, float]:
    """Held-out open-loop metrics with *fixed* flow times / noise per batch index so values are
    comparable across checkpoints and variants:
      fm_loss            masked flow-matching loss (velocity MSE)
      action_mse         masked MSE between sampled chunk (N Euler steps, fixed noise) and target
      action_mse_first   same, first action of the chunk only (what E=1 executes)
      gripper_acc        sign agreement of the sampled gripper command (valid steps)
    """
    dev = policy.device
    tot = defaultdict(float)
    n = 0
    H, A = policy.cfg.action_head.action_horizon, policy.cfg.action_head.action_dim
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        g = torch.Generator().manual_seed(seed + bi)
        B = batch["actions"].shape[0]
        t = torch.rand(B, generator=g) * 0.999
        noise = torch.randn(B, H, A, generator=g)
        loss, _ = policy.compute_loss(batch, t=t, noise=noise)
        inputs = batch["monet_inputs"].to(dev)
        state = batch["state"].to(dev)
        with policy._autocast():
            cond, _ = policy.encode(inputs, state, prev_actions=batch.get("prev_actions"),
                                    prev_valid=batch.get("prev_valid"))
            pred = policy.action_head.sample(cond.tokens.float(), cond.mask,
                                             state=state if policy.cfg.state.state_to_action_head else None,
                                             num_steps=num_steps, noise=torch.randn(B, H, A, generator=g))
        tgt, mask = batch["actions"].to(dev), batch["loss_mask"].to(dev)
        tot["fm_loss"] += float(loss)
        tot["action_mse"] += float(masked_mse(pred, tgt, mask))
        tot["action_mse_first"] += float(masked_mse(pred[:, :1], tgt[:, :1], mask[:, :1]))
        gm = mask.bool()
        agree = (torch.sign(pred[..., 6]) == torch.sign(tgt[..., 6])) & gm
        tot["gripper_acc"] += float(agree.sum() / gm.sum().clamp_min(1))
        n += 1
    return {k: v / max(1, n) for k, v in tot.items()} | {"val_batches": n}


# --------------------------------------------------------------------------- statistics
def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_diff_ci(a: List[int], b: List[int], iters: int = 10000, seed: int = 0):
    """Paired-by-index bootstrap CI for success-rate difference mean(b) - mean(a)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(iters, len(a)))
    d = b[idx].mean(1) - a[idx].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def mcnemar_exact(a: List[int], b: List[int]) -> float:
    """Exact two-sided McNemar p-value for paired binary outcomes (matched init states/seeds)."""
    from math import comb

    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


# --------------------------------------------------------------------------- aggregation
def with_derived_latency(lat: Dict[str, float]) -> Dict[str, float]:
    """Leaf components are disjoint (preprocess, monet_embed, monet_prefill, monet_latent_steps,
    conditioner, action_head); add ``monet`` (sum of monet_*) and ``total`` (sum of leaves)."""
    leaves = {k: v for k, v in lat.items() if k not in ("total", "monet")}
    out = dict(leaves)
    monet = [v for k, v in leaves.items() if k.startswith("monet_")]
    if monet:
        out["monet"] = float(sum(monet))
    out["total"] = float(sum(leaves.values()))
    return out


def aggregate_episodes(records: List[Dict], gpu_hourly_cost_usd: Optional[float] = None) -> Dict:
    """``records``: per-episode dicts written by evaluate.py."""
    out: Dict = {"n_episodes": len(records)}
    succ = [int(r["success"]) for r in records]
    k, n = sum(succ), len(succ)
    out["success_rate"] = k / n if n else float("nan")
    out["success_ci95"] = wilson_ci(k, n)
    per_task = defaultdict(lambda: [0, 0])
    per_seed = defaultdict(lambda: [0, 0])
    for r in records:
        per_task[r["task_id"]][0] += int(r["success"])
        per_task[r["task_id"]][1] += 1
        per_seed[r["seed"]][0] += int(r["success"])
        per_seed[r["seed"]][1] += 1
    out["per_task"] = {int(t): {"successes": s, "trials": m, "rate": s / m, "ci95": wilson_ci(s, m)}
                       for t, (s, m) in sorted(per_task.items())}
    seed_rates = [s / m for s, m in per_seed.values()]
    out["per_seed_rate"] = {int(sd): s / m for sd, (s, m) in sorted(per_seed.items())}
    out["seed_rate_std"] = float(np.std(seed_rates)) if seed_rates else float("nan")

    # latency (per CALL_ACTION, ms), split by component; per-sample normalisation divides
    # batched call time by the batch size.
    comp = defaultdict(list)
    comp_per_sample = defaultdict(list)
    for r in records:
        for lat, bs in zip(r["call_latency_ms"], r["batch_sizes"]):
            lat = with_derived_latency(lat)
            for kk, v in lat.items():
                comp[kk].append(v)
                comp_per_sample[kk].append(v / bs)
    out["latency_ms_per_call"] = {kk: percentiles(v) for kk, v in comp.items()}
    out["latency_ms_per_sample_amortised"] = {kk: percentiles(v) for kk, v in comp_per_sample.items()}
    calls = [r["action_calls"] for r in records]
    steps = [r["env_steps"] for r in records]
    out["action_calls_per_episode"] = float(np.mean(calls)) if calls else float("nan")
    out["env_steps_per_episode"] = float(np.mean(steps)) if steps else float("nan")
    sim = [r["sim_time_ms"] for r in records]
    out["sim_wall_s_per_episode"] = float(np.mean(sim)) / 1000 if sim else float("nan")
    # GPU-seconds: policy compute time attributed to each episode (amortised over the batch)
    gpu_s = [sum(sum(lat.values()) / bs for lat, bs in zip(r["call_latency_ms"], r["batch_sizes"])) / 1000
             for r in records]
    out["gpu_seconds_per_episode"] = float(np.mean(gpu_s)) if gpu_s else float("nan")
    out["gpu_seconds_total"] = float(np.sum(gpu_s))
    out["gpu_seconds_per_success"] = float(np.sum(gpu_s) / k) if k else float("inf")
    if gpu_hourly_cost_usd:
        out["usd_per_success"] = out["gpu_seconds_per_success"] * gpu_hourly_cost_usd / 3600
    return out


def markdown_table(summaries: Dict[str, Dict], task_names: Optional[Dict[int, str]] = None) -> str:
    """Variant comparison table (rows = variants)."""
    lines = ["| variant | success | 95% CI | GPU-s/episode | GPU-s/success | call p50 ms | call p95 ms | monet p50 ms | head p50 ms | calls/ep |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in summaries.items():
        lat = s.get("latency_ms_per_call", {})
        lines.append(
            f"| {name} | {s['success_rate']:.3f} ({sum(t['successes'] for t in s['per_task'].values())}/{s['n_episodes']}) "
            f"| [{s['success_ci95'][0]:.3f}, {s['success_ci95'][1]:.3f}] | {s['gpu_seconds_per_episode']:.2f} "
            f"| {s['gpu_seconds_per_success']:.2f} | {lat.get('total', {}).get('p50', float('nan')):.1f} "
            f"| {lat.get('total', {}).get('p95', float('nan')):.1f} | {lat.get('monet', {}).get('p50', float('nan')):.1f} "
            f"| {lat.get('action_head', {}).get('p50', float('nan')):.1f} | {s['action_calls_per_episode']:.1f} |")
    return "\n".join(lines)
