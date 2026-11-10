"""Action / state normalisation with an exact, tested inverse.

LIBERO action convention (verified in G0, see AUDIT.md):
    a[0:3]  delta end-effector position command (OSC_POSE, input range [-1, 1])
    a[3:6]  delta end-effector rotation (axis-angle) command (input range [-1, 1])
    a[6]    gripper: -1 = open, +1 = close (binary in the demonstrations)
The 6 continuous dims are normalised to [-1, 1] with the training-split q01/q99 (default)
bounds; the gripper is kept in {-1, +1} (identity transform) and binarised with sign()
before it is sent to the simulator.

LIBERO 8D proprioceptive state (LeRobot ``observation.state``; the column *names* in the
dataset metadata say "x,y,z,rx,ry,rz,rw,gripper" but the values are
eef_pos(3) + quat2axisangle(eef_quat)(3) + gripper finger qpos(2)) is z-scored.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch

GRIPPER_DIM = 6


@dataclass
class Stats:
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    q01: np.ndarray
    q99: np.ndarray

    @staticmethod
    def from_array(x: np.ndarray) -> "Stats":
        x = np.asarray(x, dtype=np.float64)
        return Stats(
            mean=x.mean(0), std=x.std(0), min=x.min(0), max=x.max(0),
            q01=np.quantile(x, 0.01, axis=0), q99=np.quantile(x, 0.99, axis=0),
        )

    def to_dict(self) -> Dict[str, list]:
        return {k: getattr(self, k).tolist() for k in ("mean", "std", "min", "max", "q01", "q99")}

    @staticmethod
    def from_dict(d: Dict[str, list]) -> "Stats":
        return Stats(**{k: np.asarray(d[k], dtype=np.float64) for k in ("mean", "std", "min", "max", "q01", "q99")})


class Normalizer:
    """Affine per-dimension normaliser; identity on ``identity_dims``."""

    def __init__(self, stats: Stats, mode: str, identity_dims=(), eps: float = 1e-6, clip: bool = True):
        self.stats, self.mode, self.eps, self.clip = stats, mode, eps, clip
        self.identity_dims = tuple(identity_dims)
        d = len(stats.mean)
        if mode == "meanstd":
            center, scale = stats.mean, np.maximum(stats.std, eps)
        elif mode in ("q99", "minmax"):
            lo, hi = (stats.q01, stats.q99) if mode == "q99" else (stats.min, stats.max)
            center, scale = (hi + lo) / 2.0, np.maximum((hi - lo) / 2.0, eps)
        elif mode == "identity":
            center, scale = np.zeros(d), np.ones(d)
        else:
            raise ValueError(mode)
        center, scale = center.copy(), scale.copy()
        for i in self.identity_dims:
            center[i], scale[i] = 0.0, 1.0
        self.center = center.astype(np.float32)
        self.scale = scale.astype(np.float32)

    def _ct(self, x):
        if isinstance(x, torch.Tensor):
            return (torch.as_tensor(self.center, device=x.device, dtype=x.dtype),
                    torch.as_tensor(self.scale, device=x.device, dtype=x.dtype))
        return self.center, self.scale

    def normalize(self, x):
        c, s = self._ct(x)
        y = (x - c) / s
        if self.clip and self.mode in ("q99", "minmax"):
            y = y.clamp(-1.0, 1.0) if isinstance(y, torch.Tensor) else np.clip(y, -1.0, 1.0)
        return y

    def unnormalize(self, y):
        c, s = self._ct(y)
        return y * s + c

    def to_dict(self) -> Dict:
        return {"mode": self.mode, "identity_dims": list(self.identity_dims), "clip": self.clip,
                "stats": self.stats.to_dict()}

    @staticmethod
    def from_dict(d: Dict) -> "Normalizer":
        return Normalizer(Stats.from_dict(d["stats"]), d["mode"], d.get("identity_dims", ()), clip=d.get("clip", True))


class ActionStateNormalizer:
    """Bundles the action and state normalisers that a checkpoint was trained with."""

    def __init__(self, action: Normalizer, state: Normalizer):
        self.action, self.state = action, state

    @staticmethod
    def fit(actions: np.ndarray, states: np.ndarray, action_mode="q99", state_mode="meanstd") -> "ActionStateNormalizer":
        a_stats, s_stats = Stats.from_array(actions), Stats.from_array(states)
        return ActionStateNormalizer(
            Normalizer(a_stats, action_mode, identity_dims=(GRIPPER_DIM,)),
            Normalizer(s_stats, state_mode, clip=False),
        )

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"action": self.action.to_dict(), "state": self.state.to_dict()}, f, indent=2)

    @staticmethod
    def load(path) -> "ActionStateNormalizer":
        with open(path) as f:
            d = json.load(f)
        return ActionStateNormalizer(Normalizer.from_dict(d["action"]), Normalizer.from_dict(d["state"]))

    # ------------------------------------------------------------------ env side
    def to_env_action(self, norm_action) -> np.ndarray:
        """Normalised model output [..., 7] -> simulator command. Gripper -> sign in {-1,+1}."""
        if isinstance(norm_action, torch.Tensor):
            norm_action = norm_action.detach().float().cpu().numpy()
        a = self.action.unnormalize(np.asarray(norm_action, dtype=np.float32))
        a = a.copy()
        a[..., :GRIPPER_DIM] = np.clip(a[..., :GRIPPER_DIM], -1.0, 1.0)
        a[..., GRIPPER_DIM] = np.where(a[..., GRIPPER_DIM] >= 0.0, 1.0, -1.0)
        return a.astype(np.float32)
