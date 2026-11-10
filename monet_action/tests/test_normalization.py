"""Critical test 7: action normalisation / unnormalisation and gripper handling."""
import numpy as np
import torch

from monet_action.normalization import GRIPPER_DIM, ActionStateNormalizer


def _fit(seed=0):
    rng = np.random.default_rng(seed)
    a = np.concatenate([rng.normal(0, 0.4, (5000, 3)), rng.normal(0, 0.05, (5000, 3)),
                        rng.choice([-1.0, 1.0], (5000, 1))], 1).astype(np.float32)
    s = rng.normal(0, 1, (5000, 8)).astype(np.float32)
    return ActionStateNormalizer.fit(a, s), a, s


def test_roundtrip_and_ranges(tmp_path):
    n, a, s = _fit()
    na = n.action.normalize(a)
    assert na[:, :6].min() >= -1 and na[:, :6].max() <= 1
    inside = (a[:, :6] >= n.action.stats.q01[:6]) & (a[:, :6] <= n.action.stats.q99[:6])
    rt = n.action.unnormalize(na)
    assert np.abs(rt - a)[:, :6][inside].max() < 1e-5
    assert set(np.unique(na[:, GRIPPER_DIM]).tolist()) == {-1.0, 1.0}  # gripper untouched
    ns = n.state.normalize(s)
    assert abs(ns.mean()) < 0.05 and abs(ns.std() - 1) < 0.05
    np.testing.assert_allclose(n.state.unnormalize(ns), s, atol=1e-5)
    n.save(tmp_path / "n.json")
    n2 = type(n).load(tmp_path / "n.json")
    np.testing.assert_allclose(n2.action.normalize(a), na)
    t = torch.from_numpy(a)
    np.testing.assert_allclose(n.action.normalize(t).numpy(), na, atol=1e-6)


def test_env_action_conversion():
    n, a, _ = _fit()
    norm = np.array([[0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.3], [2.0, -3.0, 0.0, 0.0, 0.0, 0.0, -0.01]], np.float32)
    env = n.to_env_action(norm)
    assert env.shape == (2, 7)
    assert env[0, 6] == 1.0 and env[1, 6] == -1.0  # sign: +1 close, -1 open
    assert np.all(np.abs(env[:, :6]) <= 1.0)  # OSC input range
    np.testing.assert_allclose(env[0, :6], np.clip(n.action.unnormalize(norm)[0, :6], -1, 1), atol=1e-6)
