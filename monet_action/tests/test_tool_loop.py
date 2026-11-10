"""Critical tests 9-10: CALL_ACTION / TOOL_RESULT, env stepping, observation refresh, no oracle data.

``CountingEnv`` below is a *synthetic* stand-in used only to test the runtime's control flow;
it is never reported as a LIBERO rollout. The real-simulator test is marked ``sim``.
"""
import dataclasses

import numpy as np
import pytest
import torch

from monet_action.config import load_config
from monet_action.libero_env import EnvStep, PolicyObservation
from monet_action.monet_loader import build_tiny_monet
from monet_action.normalization import ActionStateNormalizer
from monet_action.policy import MonetActionPolicy
from monet_action.tool_runtime import ActionTool, CallActionRequest, CallActionResponse, run_episodes_lockstep

from conftest import sim_available

pytestmark = pytest.mark.tokenizer


class CountingEnv:
    """Synthetic env: state integrates the commanded xyz; succeeds after `succeed_at` steps."""

    def __init__(self, succeed_at=None, size=64):
        self.succeed_at, self.size, self.t, self.state, self.calls = succeed_at, size, 0, np.zeros(8, np.float32), []

    def _obs(self):
        img = np.full((self.size, self.size, 3), self.t % 255, np.uint8)
        return PolicyObservation({"image": img, "wrist_image": img.copy()}, self.state.copy(), "pick up the bowl")

    def reset(self, init_state_idx, num_wait_steps=10):
        self.t, self.state = 0, np.full(8, float(init_state_idx), np.float32)
        return self._obs()

    def step(self, action):
        self.calls.append(np.array(action))
        self.t += 1
        self.state[:3] += 0.01 * np.asarray(action)[:3]
        done = self.succeed_at is not None and self.t >= self.succeed_at
        return EnvStep(self._obs(), float(done), done, {"sim_success": done})


def _policy(cfg_path, processor):
    cfg = load_config(cfg_path)
    rng = np.random.default_rng(0)
    a = np.concatenate([rng.normal(0, 0.3, (500, 6)), rng.choice([-1.0, 1.0], (500, 1))], 1).astype(np.float32)
    norm = ActionStateNormalizer.fit(a, rng.normal(0, 1, (500, 8)).astype(np.float32))
    return MonetActionPolicy(cfg, build_tiny_monet(seed=0), processor, norm).eval(), cfg


def test_policy_observation_has_no_oracle_fields():
    """Only camera images, proprioception, the instruction and the agent's own previously executed
    commands (TOOL_RESULT) may reach the policy - never simulator object poses or success flags."""
    names = {f.name for f in dataclasses.fields(PolicyObservation)}
    assert names == {"images", "state", "instruction", "prev_actions"}


def test_call_action_schema_validation(cfg_path, processor):
    pol, cfg = _policy(cfg_path, processor)
    env = CountingEnv()
    req = CallActionRequest(env.reset(0), num_integration_steps=4, execution_prefix=1)
    req.validate(8)
    bad = CallActionRequest(PolicyObservation({"image": np.zeros((4, 4), np.uint8)}, np.zeros(8, np.float32), "x"), 4, 1)
    with pytest.raises(ValueError):
        bad.validate(8)
    with pytest.raises(ValueError):
        CallActionRequest(env.reset(0), 4, 9).validate(8)
    tool = ActionTool(pol, 8, generator=torch.Generator().manual_seed(0))
    (resp,) = tool([req])
    assert isinstance(resp, CallActionResponse) and resp.env_chunk.shape == (8, 7)
    assert set(np.unique(resp.env_chunk[:, 6]).tolist()) <= {-1.0, 1.0}
    assert {"preprocess", "monet_embed", "monet_prefill", "monet_latent_steps", "conditioner",
            "action_head"} <= set(resp.timings_ms)


@pytest.mark.parametrize("E", [1, 2])
def test_lockstep_loop_executes_prefix_and_refreshes_observation(cfg_path, processor, E):
    pol, cfg = _policy(cfg_path, processor)
    envs = [CountingEnv(succeed_at=5), CountingEnv(succeed_at=None)]
    tool = ActionTool(pol, 8, generator=torch.Generator().manual_seed(0))
    recs = run_episodes_lockstep(envs, [0, 1], tool, max_steps=7, execution_prefix=E, num_integration_steps=2,
                                 task_ids=[3, 3])
    assert recs[0].success and recs[0].env_steps == 5
    assert not recs[1].success and recs[1].env_steps == 7
    assert recs[1].action_calls == int(np.ceil(7 / E))
    assert len(envs[1].calls) == 7  # env.step really called once per executed action
    tr = recs[1].trace
    assert all(len(c["applied"]) <= E for c in tr)
    states = [c["state_after"] for c in tr]
    assert len({tuple(s) for s in states}) > 1  # observation refreshed between calls
    assert recs[0].batch_sizes[0] == 2 and recs[1].batch_sizes[-1] == 1  # batched until env 0 finished


def test_latent_shuffle_changes_prediction(cfg_path, processor):
    """Policy output depends on the latent content/order (sensitivity test, not a causal claim)."""
    pol, _ = _policy(cfg_path, processor)
    obs = [CountingEnv().reset(0)]
    noise = torch.randn(1, 8, 7, generator=torch.Generator().manual_seed(0))
    base = pol.predict(obs, num_steps=2, noise=noise).norm_actions
    for spec in ("zero", "mask", "shuffle_order"):
        alt = pol.predict(obs, num_steps=2, noise=noise, intervention=spec,
                          generator=torch.Generator().manual_seed(1)).norm_actions
        assert (alt - base).abs().max() > 1e-6, spec
    again = pol.predict(obs, num_steps=2, noise=noise).norm_actions
    torch.testing.assert_close(base, again)


@pytest.mark.sim
@pytest.mark.skipif(not sim_available(), reason="LIBERO/robosuite/MuJoCo not installed")
def test_real_libero_call_action_cycle(cfg_path, processor):
    """One real LIBERO reset -> CALL_ACTION -> env.step -> TOOL_RESULT -> CALL_ACTION cycle."""
    from monet_action.libero_env import LiberoEnv

    pol, cfg = _policy(cfg_path, processor)
    env = LiberoEnv("libero_spatial", 0, camera_size=128, seed=0)
    try:
        tool = ActionTool(pol, 8, generator=torch.Generator().manual_seed(0))
        recs = run_episodes_lockstep([env], [0], tool, max_steps=4, execution_prefix=2, num_integration_steps=2,
                                     num_wait_steps=2, task_ids=[0])
        r = recs[0]
        assert r.env_steps == 4 and r.action_calls == 2
        s0, s1 = np.array(r.trace[0]["state_after"]), np.array(r.trace[1]["state_after"])
        assert not np.allclose(s0, s1)
        assert r.sim_time_ms > 0
    finally:
        env.close()
