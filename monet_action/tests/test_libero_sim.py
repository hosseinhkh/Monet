"""LIBERO / robosuite / MuJoCo checks (marker ``sim``): controller, resets, image/state
conventions and an end-to-end replay of a recorded LIBERO demonstration.

Raw demo file (optional, ~500 MB): scripts/download_libero_raw.sh, or set MONET_ACTION_LIBERO_RAW.
"""
import glob
import os
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT, sim_available

pytestmark = [pytest.mark.sim, pytest.mark.skipif(not sim_available(), reason="LIBERO/robosuite/MuJoCo not installed")]

TASK0_RAW = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"


def _raw_file():
    p = os.environ.get("MONET_ACTION_LIBERO_RAW")
    cands = [p] if p else glob.glob(str(ROOT / "data" / "libero_raw" / "libero_spatial" / TASK0_RAW))
    cands = [c for c in cands if c and Path(c).exists()]
    if not cands:
        pytest.skip("raw LIBERO demo file not found (scripts/download_libero_raw.sh)")
    return cands[0]


@pytest.fixture(scope="module")
def env0():
    from monet_action.libero_env import LiberoEnv

    e = LiberoEnv("libero_spatial", 0, camera_size=128, seed=0)
    yield e
    e.close()


def test_suite_and_controller(env0):
    from monet_action.libero_env import libero_task_table

    table = libero_task_table("libero_spatial")
    assert len(table) == 10 and all(t[2].startswith("pick up the black bowl") for t in table)
    robot = env0.env.env.robots[0]
    assert type(robot.controller).__name__ == "OperationalSpaceController"
    assert env0.env.env.control_freq == 20
    assert np.allclose(robot.controller.input_max, 1) and np.allclose(robot.controller.input_min, -1)
    assert len(env0.init_states) == 50 and env0.max_steps == 220


def test_reset_is_deterministic_and_init_states_differ(env0):
    a = env0.reset(3, num_wait_steps=2)
    b = env0.reset(3, num_wait_steps=2)
    np.testing.assert_allclose(a.state, b.state, atol=1e-8)
    assert np.array_equal(a.images["image"], b.images["image"])
    c = env0.reset(4, num_wait_steps=2)
    assert not np.array_equal(a.images["image"], c.images["image"])  # different object placement
    assert not env0.check_success()  # success flag only for scoring, false at the start
    assert a.prev_actions is None


def test_step_refreshes_observation(env0):
    o = env0.reset(0, num_wait_steps=1)
    st = env0.step(np.array([0.5, 0, 0, 0, 0, 0, -1], np.float32))
    assert not np.allclose(st.observation.state[:3], o.state[:3])
    assert set(st.diagnostics) >= {"sim_success", "t"}


def test_image_orientation_matches_dataset(data_root):
    from monet_action.libero_dataset import LeRobotLiberoFrames
    from monet_action.libero_env import LiberoEnv, libero_task_table

    frames = LeRobotLiberoFrames(data_root, file_limit=1)
    e0 = frames.episodes[0]
    tid = [t for t, _, lang in libero_task_table() if lang == e0.instruction][0]
    env = LiberoEnv("libero_spatial", tid, camera_size=256, seed=0)
    try:
        obs = env.reset(0)
    finally:
        env.close()
    for key in ("image", "wrist_image"):
        sim = obs.images[key].astype(np.float32).ravel()
        flipped_back = obs.images[key][::-1, ::-1].astype(np.float32).ravel()
        ds = frames.image(key, e0.start).astype(np.float32).ravel()
        assert np.corrcoef(sim, ds)[0, 1] > 0.8, key
        assert np.corrcoef(flipped_back, ds)[0, 1] < 0.3, key


def test_state_conversion_matches_raw_demo(env0):
    import h5py

    from monet_action.libero_env import raw_obs_to_policy

    with h5py.File(_raw_file(), "r") as f:
        d = f["data"]["demo_0"]
        states, ee = d["states"][:], d["obs"]["ee_states"][:]
    env0.env.reset()
    obs = env0.env.set_init_state(states[0])
    s = raw_obs_to_policy(obs, env0.instruction).state
    np.testing.assert_allclose(s[:3], ee[0, :3], atol=5e-3)  # eef position
    np.testing.assert_allclose(s[3:6], ee[0, 3:6], atol=2e-2)  # axis-angle orientation


@pytest.mark.slow
def test_recorded_demo_replays_to_success(env0):
    """End-to-end action convention check: raw demo actions (delta EEF, gripper -1 open / +1 close)
    replayed from the recorded initial state must solve the task in our simulator build."""
    import h5py

    with h5py.File(_raw_file(), "r") as f:
        d = f["data"]["demo_0"]
        states, actions = d["states"][:], d["actions"][:]
    env0.env.reset()
    env0.env.set_init_state(states[0])
    done = False
    for a in actions:
        _, _, done, _ = env0.env.step(a.tolist())
        if done:
            break
    assert done, "replayed demonstration did not succeed - check controller / gripper conventions"


def _blockiness(a):
    g = np.abs(np.diff(a.astype(np.float32).mean(-1), axis=1))
    c = np.arange(g.shape[1])
    return g[:, (c % 8) == 7].mean() / g[:, (c % 8) != 7].mean()


def test_eval_jpeg_roundtrip_matches_training_image_statistics(data_root):
    """Training frames carry JPEG 8x8 block artefacts; evaluation frames get a JPEG round-trip
    (eval.jpeg_quality) so the policy sees the same image statistics."""
    from monet_action.config import EvalConfig
    from monet_action.libero_dataset import LeRobotLiberoFrames
    from monet_action.libero_env import LiberoEnv, jpeg_roundtrip

    frames = LeRobotLiberoFrames(data_root, file_limit=1)
    ds = np.mean([_blockiness(frames.image("image", e.start)) for e in frames.episodes])
    q = EvalConfig().jpeg_quality
    env = LiberoEnv("libero_spatial", 6, camera_size=256, seed=0)
    try:
        raw = [env.reset(i, num_wait_steps=1).images["image"] for i in (0, 1)]
    finally:
        env.close()
    b_raw = np.mean([_blockiness(im) for im in raw])
    b_jpg = np.mean([_blockiness(jpeg_roundtrip(im, q)) for im in raw])
    assert abs(b_jpg - ds) < abs(b_raw - ds), (ds, b_raw, b_jpg)
    out = jpeg_roundtrip(raw[0], q)
    assert out.shape == raw[0].shape and out.dtype == np.uint8 and jpeg_roundtrip(raw[0], None) is raw[0]
