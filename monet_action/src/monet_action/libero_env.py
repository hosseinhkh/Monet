"""Thin LIBERO wrapper producing *policy-ready* observations.

Policy inputs (and nothing else):
  images["image"]       agent-view RGB, rotated 180 deg ([::-1, ::-1]) to match the LeRobot /
                        OpenVLA-regenerated demonstrations (verified in G0 by image correlation)
  images["wrist_image"] eye-in-hand RGB, same rotation
  state (8,)            eef_pos(3) + quat2axisangle(eef_quat)(3) + gripper_qpos(2), float32
Simulator success flags, object poses and the raw MuJoCo state are returned separately under
``diagnostics`` for evaluation/logging only and must never be passed to the policy (enforced by
:class:`PolicyObservation` having no field for them; see tests/test_tool_loop.py).
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

LIBERO_DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)  # no motion, gripper open
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def _ensure_gl_backend() -> None:
    if "MUJOCO_GL" not in os.environ:
        try:
            import torch

            os.environ["MUJOCO_GL"] = "egl" if torch.cuda.is_available() else "osmesa"
        except Exception:
            os.environ["MUJOCO_GL"] = "osmesa"
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """robosuite transform_utils.quat2axisangle (xyzw)."""
    q = np.array(quat, dtype=np.float64).copy()
    q[3] = min(1.0, max(-1.0, q[3]))
    den = math.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


@dataclass
class PolicyObservation:
    images: Dict[str, np.ndarray]
    state: np.ndarray
    instruction: str
    # The agent's own previously executed commands (newest last), filled by the tool runtime from
    # TOOL_RESULT. Not privileged: the policy issued them itself. None at episode start.
    prev_actions: Optional[np.ndarray] = None


@dataclass
class EnvStep:
    observation: PolicyObservation
    reward: float
    done: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def raw_obs_to_policy(obs: Dict[str, np.ndarray], instruction: str) -> PolicyObservation:
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]]).astype(np.float32)
    return PolicyObservation(images={"image": img, "wrist_image": wrist}, state=state, instruction=instruction)


def libero_task_table(suite_name: str = "libero_spatial"):
    _ensure_gl_backend()
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[suite_name]()
    return [(i, suite.get_task(i).name, suite.get_task(i).language) for i in range(suite.n_tasks)]


class LiberoEnv:
    def __init__(self, suite_name: str, task_id: int, camera_size: int = 256, seed: int = 0):
        _ensure_gl_backend()
        import torch
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.suite_name, self.task_id = suite_name, task_id
        suite = benchmark.get_benchmark_dict()[suite_name]()
        self.task = suite.get_task(task_id)
        self.instruction = self.task.language
        bddl = os.path.join(get_libero_path("bddl_files"), self.task.problem_folder, self.task.bddl_file)
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=camera_size, camera_widths=camera_size)
        self.env.seed(seed)
        # Fixed benchmark initial states shipped with the pinned LIBERO commit (numpy pickle;
        # torch>=2.6 needs weights_only=False for these trusted upstream files).
        p = os.path.join(get_libero_path("init_states"), self.task.problem_folder, self.task.init_states_file)
        self.init_states = torch.load(p, weights_only=False)
        self.max_steps = MAX_STEPS.get(suite_name, 300)
        self.t = 0

    def reset(self, init_state_idx: int, num_wait_steps: int = 10, reset_seed: Optional[int] = None) -> PolicyObservation:
        """Reset to a fixed benchmark init state.

        LIBERO hard-resets (reloads the scene XML) on every ``reset()`` and re-samples some fixture
        textures from Python's *global* ``random``/``numpy`` RNGs, so without seeding, the same init
        state renders differently depending on how many resets happened before (verified in G0:
        identical MuJoCo state, ~12-16% of pixels differ). The RNGs are therefore seeded from
        (task, init state) - matched across variants and seeds - and restored afterwards.
        """
        seed = reset_seed if reset_seed is not None else 1_000_003 * self.task_id + int(init_state_idx)
        py_state, np_state = random.getstate(), np.random.get_state()
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        try:
            self.env.reset()
            obs = self.env.set_init_state(self.init_states[init_state_idx % len(self.init_states)])
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
        for _ in range(num_wait_steps):  # let objects settle (standard LIBERO/OpenVLA protocol)
            obs, _, _, _ = self.env.step(LIBERO_DUMMY_ACTION.tolist())
        self.t = 0
        self._last_raw = obs
        return raw_obs_to_policy(obs, self.instruction)

    def step(self, action: np.ndarray) -> EnvStep:
        action = np.asarray(action, dtype=np.float32).reshape(7)
        obs, reward, done, info = self.env.step(action.tolist())
        self.t += 1
        self._last_raw = obs
        diag = {"sim_success": bool(done), "t": self.t}
        return EnvStep(raw_obs_to_policy(obs, self.instruction), float(reward), bool(done), diag)

    def check_success(self) -> bool:
        return bool(self.env.check_success())

    def render_frame(self) -> np.ndarray:
        return np.ascontiguousarray(self._last_raw["agentview_image"][::-1, ::-1])

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
