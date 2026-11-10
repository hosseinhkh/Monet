"""TOOL_RESULT context, evaluation protocol helpers, bypass diagnostics and the D2 retention term."""
import numpy as np
import pytest
import torch

from monet_action.config import EvalConfig, load_config
from monet_action.libero_dataset import EpisodeInfo, LiberoWindowDataset
from monet_action.monet_loader import build_tiny_monet
from monet_action.normalization import ActionStateNormalizer
from monet_action.policy import MonetActionPolicy
from monet_action.tool_runtime import ActionTool, EpisodeRecord, auto_failure_tags, run_episodes_lockstep

from conftest import INSTR, ROOT, random_views
from test_tool_loop import CountingEnv


class FakeFrames:
    camera_keys = ["image"]
    state = np.zeros((20, 8), np.float32)
    action = np.repeat(np.arange(20, dtype=np.float32)[:, None], 7, 1)

    def image(self, k, row):
        return np.zeros((4, 4, 3), np.uint8)


def test_dataset_previous_actions_respect_episode_boundaries():
    eps = [EpisodeInfo(0, 0, "a", 0, 12), EpisodeInfo(1, 0, "a", 12, 8)]
    ds = LiberoWindowDataset(FakeFrames(), eps, horizon=8, num_prev_actions=2)
    s0 = ds[0]
    assert s0["prev_valid"].tolist() == [0, 0] and not s0["prev_actions"].any()
    s1 = ds[1]
    assert s1["prev_valid"].tolist() == [0, 1] and s1["prev_actions"][:, 0].tolist() == [0, 0]
    s5 = ds[5]
    assert s5["prev_valid"].tolist() == [1, 1] and s5["prev_actions"][:, 0].tolist() == [3, 4]
    first_of_ep1 = ds[12]  # must not see episode 0's last actions
    assert first_of_ep1["prev_valid"].tolist() == [0, 0] and not first_of_ep1["prev_actions"].any()


def _policy(cfg_path, processor, overrides=()):
    cfg = load_config(cfg_path, list(overrides))
    rng = np.random.default_rng(0)
    a = np.concatenate([rng.normal(0, 0.3, (500, 6)), rng.choice([-1.0, 1.0], (500, 1))], 1).astype(np.float32)
    norm = ActionStateNormalizer.fit(a, rng.normal(0, 1, (500, 8)).astype(np.float32))
    return MonetActionPolicy(cfg, build_tiny_monet(seed=0), processor, norm).eval(), cfg


@pytest.mark.tokenizer
def test_prediction_depends_on_tool_result(cfg_path, processor):
    pol, _ = _policy(cfg_path, processor)
    obs = CountingEnv().reset(0)
    noise = torch.randn(1, 8, 7, generator=torch.Generator().manual_seed(0))
    base = pol.predict([obs], num_steps=2, noise=noise).norm_actions
    obs.prev_actions = np.array([[0.5, -0.5, 0.2, 0, 0, 0, 1.0]], np.float32)
    with_prev = pol.predict([obs], num_steps=2, noise=noise).norm_actions
    assert (with_prev - base).abs().max() > 1e-6
    prev, valid = pol.prev_actions_from_observations([obs, CountingEnv().reset(0)])
    assert valid.tolist() == [[1.0], [0.0]] and prev.shape == (2, 1, 7)


@pytest.mark.tokenizer
def test_tool_result_dropout_only_in_training(cfg_path, processor):
    pol, _ = _policy(cfg_path, processor, ["tool_result.dropout=1.0"])
    prev, valid = torch.ones(3, 1, 7), torch.ones(3, 1)
    pol.eval()
    assert pol.tool_result_features(prev, valid, 3)[:, -1].tolist() == [1, 1, 1]
    pol.train()
    f = pol.tool_result_features(prev, valid, 3)
    assert f.abs().sum() == 0  # hidden: values and validity zeroed


@pytest.mark.tokenizer
def test_runtime_feeds_executed_actions_back(cfg_path, processor):
    pol, _ = _policy(cfg_path, processor)
    seen = []

    class SpyTool(ActionTool):
        def __call__(self, requests, bank_keys=None):
            seen.append([None if r.observation.prev_actions is None else r.observation.prev_actions.copy()
                         for r in requests])
            return super().__call__(requests, bank_keys)

    env = CountingEnv()
    tool = SpyTool(pol, 8, generator=torch.Generator().manual_seed(0))
    run_episodes_lockstep([env], [0], tool, max_steps=4, execution_prefix=2, num_integration_steps=2)
    assert seen[0][0] is None  # episode start: no TOOL_RESULT yet
    np.testing.assert_allclose(seen[1][0][-2:], np.stack(env.calls[:2]))  # the two executed actions


@pytest.mark.tokenizer
def test_bypass_interventions_change_prediction(cfg_path, processor):
    pol, _ = _policy(cfg_path, processor)
    obs = [CountingEnv().reset(0)]
    noise = torch.randn(1, 8, 7, generator=torch.Generator().manual_seed(0))
    base = pol.predict(obs, num_steps=2, noise=noise).norm_actions
    for spec in ("mask_vl", "mask_all"):
        alt = pol.predict(obs, num_steps=2, noise=noise, intervention=spec).norm_actions
        assert (alt - base).abs().max() > 1e-6, spec


@pytest.mark.tokenizer
def test_latent_retention_regulariser(cfg_path, processor):
    pol, _ = _policy(cfg_path, processor, ["monet.lora_enabled=true", "monet.lora_layers=[0,1]",
                                          "train.latent_retention_weight=1.0"])
    pol.train()
    g = torch.Generator().manual_seed(0)
    batch = {"monet_inputs": pol.prompt_builder.build(random_views(2), INSTR), "state": torch.randn(2, 8, generator=g),
             "actions": torch.rand(2, 8, 7, generator=g) * 2 - 1, "loss_mask": torch.ones(2, 8)}
    _, m0 = pol.compute_loss(batch)
    assert m0["retention_loss"] < 1e-10  # LoRA B=0 at init: identical to frozen Monet
    opt = torch.optim.SGD([p for p in pol.parameters() if p.requires_grad], lr=0.5)
    for _ in range(2):
        opt.zero_grad()
        loss, _ = pol.compute_loss(batch)
        loss.backward()
        opt.step()
    _, m1 = pol.compute_loss(batch)
    assert m1["retention_loss"] > 0
    assert all(p.requires_grad for n, p in pol.named_parameters() if "lora_" in n)  # adapters re-enabled


def test_final_and_development_init_states_are_disjoint():
    from monet_action.evaluate import init_state_indices

    final = EvalConfig(seeds=[0, 1, 2], episodes_per_task=10, init_state_policy="seed_offset")
    idx = [init_state_indices(final, i) for i in range(3)]
    assert sorted(sum(idx, [])) == list(range(30))
    smoke = load_config(str(ROOT / "configs" / "eval_smoke.yaml")).eval
    dev = init_state_indices(smoke, 0)
    assert min(dev) >= 30 and max(dev) < 50
    tcfg = load_config(str(ROOT / "configs" / "base.yaml")).train
    assert tcfg.sim_eval_init_offset >= 30


def _rec(actions, states, success=False, steps=None):
    tr = [{"applied": [a], "state_after": s} for a, s in zip(actions, states)]
    return EpisodeRecord(0, 0, 0, success, steps or len(actions), len(actions), 0.0, [], [], tr)


def test_failure_tags():
    still = [[0.0, 0, 0, 0, 0, 0, -1]] * 60
    st = [[0.1, 0.2, 1.0] + [0] * 5] * 60
    tags = auto_failure_tags(_rec(still, st, steps=220), 220)
    assert {"timeout", "never_closed_gripper", "stalled"} <= set(tags)
    osc = [[1.0 if i % 2 else -1.0, 0, 0, 0, 0, 0, 1 if i % 3 else -1] for i in range(40)]
    pos = [[0.01 * (i % 2), 0, 1.0] + [0] * 5 for i in range(40)]
    tags = auto_failure_tags(_rec(osc, pos), 220)
    assert {"oscillation", "gripper_chatter", "released_after_grasp"} <= set(tags)
    assert auto_failure_tags(_rec(osc, pos, success=True), 220) == []
