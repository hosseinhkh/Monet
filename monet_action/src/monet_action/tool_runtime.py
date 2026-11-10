"""Typed CALL_ACTION / TOOL_RESULT runtime (fixed dispatch, Phase 1).

One decision cycle:
    observation --(Monet K-step latent reasoning)--> CALL_ACTION(request)
      -> DiT samples an [H, 7] chunk -> the first E actions are executed with env.step(...)
      -> TOOL_RESULT(fresh observation, applied actions, reward, done) -> next cycle
There is no learned router: CALL_ACTION is always dispatched (OBSERVE/THINK/PLAN routing is
Phase 3). The environment is stepped for real; environment execution is not differentiated.
Several environments can be advanced in lock-step so that policy calls are batched.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np
import torch

from .libero_env import EnvStep, PolicyObservation


# ------------------------------------------------------------------------- schemas
@dataclass
class CallActionRequest:
    """Input schema of the action tool."""
    observation: PolicyObservation
    num_integration_steps: int
    execution_prefix: int
    request_id: str = ""

    def validate(self, horizon: int) -> None:
        o = self.observation
        if o.state.shape != (8,) or not np.all(np.isfinite(o.state)):
            raise ValueError(f"CALL_ACTION: invalid state {o.state}")
        for k, im in o.images.items():
            if im.ndim != 3 or im.shape[2] != 3 or im.dtype != np.uint8:
                raise ValueError(f"CALL_ACTION: invalid image {k} {im.shape} {im.dtype}")
        if not 1 <= self.execution_prefix <= horizon:
            raise ValueError("execution_prefix must be in [1, H]")
        if self.num_integration_steps < 1:
            raise ValueError("num_integration_steps must be >= 1")


@dataclass
class CallActionResponse:
    """Output schema of the action tool."""
    norm_chunk: np.ndarray  # [H, 7] normalised
    env_chunk: np.ndarray  # [H, 7] simulator commands
    timings_ms: Dict[str, float]

    def validate(self, horizon: int) -> None:
        if self.env_chunk.shape != (horizon, 7) or not np.all(np.isfinite(self.env_chunk)):
            raise ValueError(f"CALL_ACTION response invalid: {self.env_chunk.shape}")
        if np.any(np.abs(self.env_chunk[:, :6]) > 1.0 + 1e-6):
            raise ValueError("continuous commands outside the OSC input range [-1, 1]")
        if not np.all(np.isin(self.env_chunk[:, 6], (-1.0, 1.0))):
            raise ValueError("gripper command must be -1 (open) or +1 (close)")


@dataclass
class ToolResult:
    """What the action tool returns to the reasoning loop after execution."""
    observation: PolicyObservation  # fresh observation after the executed prefix
    applied_actions: np.ndarray  # [E', 7] actually executed (E' < E if the episode ended)
    reward: float
    done: bool
    env_steps: int  # total env steps in the episode so far
    sim_time_ms: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)  # eval-only, never fed back


class StepEnv(Protocol):
    def reset(self, init_state_idx: int, num_wait_steps: int = 10) -> PolicyObservation: ...
    def step(self, action: np.ndarray) -> EnvStep: ...


# ------------------------------------------------------------------------- runtime
class ActionTool:
    """The callable action tool: wraps the policy's reasoning + DiT sampling."""

    def __init__(self, policy, horizon: int, generator: Optional[torch.Generator] = None,
                 intervention: Optional[str] = None):
        self.policy, self.H, self.generator, self.intervention = policy, horizon, generator, intervention
        self.num_calls = 0

    def __call__(self, requests: List[CallActionRequest], bank_keys=None) -> List[CallActionResponse]:
        for r in requests:
            r.validate(self.H)
        steps = requests[0].num_integration_steps
        pred = self.policy.predict([r.observation for r in requests], num_steps=steps, generator=self.generator,
                                   intervention=self.intervention, bank_keys=bank_keys)
        self.num_calls += 1
        # bank the *unperturbed* latents (keyed by task) for the D3 "swap" intervention
        if pred.latents is not None and pred.latents.shape[1] > 0:
            self.policy.latent_bank.add(pred.latents, bank_keys or [None] * len(requests))
        out = []
        for i in range(len(requests)):
            resp = CallActionResponse(pred.norm_actions[i].cpu().numpy(), pred.env_actions[i], dict(pred.timings_ms))
            resp.validate(self.H)
            out.append(resp)
        return out


@dataclass
class EpisodeRecord:
    task_id: int
    init_state_idx: int
    seed: int
    success: bool
    env_steps: int
    action_calls: int
    sim_time_ms: float
    call_latency_ms: List[Dict[str, float]]
    batch_sizes: List[int]
    trace: List[Dict[str, Any]]
    frames: Optional[List[np.ndarray]] = None


def run_episodes_lockstep(envs: List[StepEnv], init_idxs: List[int], tool: ActionTool, *, max_steps: int,
                          execution_prefix: int, num_integration_steps: int, num_wait_steps: int = 10,
                          record_frames: Optional[List[bool]] = None, task_ids: Optional[List[int]] = None,
                          seed: int = 0, trace_every_call: bool = True,
                          on_call: Optional[Callable[[int, List[CallActionResponse]], None]] = None) -> List[EpisodeRecord]:
    """Run one episode per env, batching CALL_ACTION across the envs that are still active."""
    n = len(envs)
    task_ids = task_ids or [-1] * n
    record_frames = record_frames or [False] * n
    obs: List[PolicyObservation] = []
    t0 = time.perf_counter()
    for e, idx in zip(envs, init_idxs):
        obs.append(e.reset(idx, num_wait_steps))
    reset_ms = (time.perf_counter() - t0) * 1000.0
    recs = [EpisodeRecord(task_ids[i], init_idxs[i], seed, False, 0, 0, reset_ms / n, [], [], [],
                          [obs[i].images["image"].copy()] if record_frames[i] else None) for i in range(n)]
    active = list(range(n))
    while active:
        reqs = [CallActionRequest(obs[i], num_integration_steps, execution_prefix, f"{task_ids[i]}:{init_idxs[i]}")
                for i in active]
        responses = tool(reqs, bank_keys=[task_ids[i] for i in active])
        if on_call:
            on_call(len(active), responses)
        still = []
        for i, resp in zip(active, responses):
            rec = recs[i]
            rec.action_calls += 1
            rec.call_latency_ms.append(resp.timings_ms)
            rec.batch_sizes.append(len(active))
            applied, done, reward, sim_ms = [], False, 0.0, 0.0
            for a in resp.env_chunk[:execution_prefix]:
                ts = time.perf_counter()
                st = envs[i].step(a)
                sim_ms += (time.perf_counter() - ts) * 1000.0
                applied.append(a)
                rec.env_steps += 1
                obs[i] = st.observation
                reward, done = st.reward, st.done
                if record_frames[i]:
                    rec.frames.append(st.observation.images["image"].copy())
                if done or rec.env_steps >= max_steps:
                    break
            rec.sim_time_ms += sim_ms
            result = ToolResult(obs[i], np.stack(applied), reward, done, rec.env_steps, sim_ms, {"sim_success": done})
            if trace_every_call:
                rec.trace.append({"call": rec.action_calls, "env_steps": result.env_steps,
                                  "applied": result.applied_actions.round(4).tolist(), "reward": result.reward,
                                  "done": result.done, "state_after": result.observation.state.round(4).tolist(),
                                  "latency_ms": {k: round(v, 3) for k, v in resp.timings_ms.items()},
                                  "sim_ms": round(sim_ms, 2)})
            if done:
                rec.success = True
            elif rec.env_steps < max_steps:
                still.append(i)
        active = still
    return recs
