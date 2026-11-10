"""Dataset / convention checks on the real LeRobot LIBERO-Spatial files (G0 data gate)."""
import numpy as np
import pytest
import torch

from monet_action.libero_dataset import Collator, LeRobotLiberoFrames, LiberoWindowDataset, split_episodes
from monet_action.normalization import ActionStateNormalizer


@pytest.fixture(scope="module")
def frames(data_root):
    return LeRobotLiberoFrames(data_root, file_limit=1)


def test_schema_and_shapes(frames):
    assert frames.state.shape[1] == 8 and frames.action.shape[1] == 7
    assert len(frames.episodes) >= 2
    e = frames.episodes[0]
    img = frames.image("image", e.start)
    wr = frames.image("wrist_image", e.start)
    assert img.shape == (256, 256, 3) and wr.shape == (256, 256, 3) and img.dtype == np.uint8
    assert isinstance(e.instruction, str) and e.instruction.startswith("pick up the black bowl")
    for ep in frames.episodes:  # contiguous frames, one task per episode
        sl = slice(ep.start, ep.start + ep.length)
        assert np.array_equal(frames.frame_index[sl], np.arange(ep.length))
        assert len(np.unique(frames.task_index[sl])) == 1 and len(np.unique(frames.episode_index[sl])) == 1


def test_gripper_and_action_conventions(frames):
    ep, st, ac = frames.episode_index, frames.state, frames.action
    assert set(np.unique(ac[:, 6]).tolist()) == {-1.0, 1.0}
    same = np.r_[ep[1:] == ep[:-1], False]
    dq = np.r_[np.diff(st[:, 6]), 0]
    assert dq[(ac[:, 6] == 1) & same].mean() < 0 < dq[(ac[:, 6] == -1) & same].mean()  # +1 closes
    dpos = np.r_[st[1:, :3] - st[:-1, :3], np.zeros((1, 3))]
    for i in range(3):  # relative EEF-delta control in the base frame
        assert np.corrcoef(ac[same, i], dpos[same, i])[0, 1] > 0.8
    aa = np.linalg.norm(st[:, 3:6], axis=1)
    assert 2.5 < aa.min() and aa.max() < 3.8  # axis-angle (gripper pointing down ~ pi), not a quaternion
    assert np.all(st[:, 6] >= -1e-3) and np.all(st[:, 7] <= 1e-3)  # two finger joints


def test_episode_disjoint_split(frames):
    tr, va = split_episodes(frames.episodes, 0.3, seed=0)
    assert not ({e.episode_index for e in tr} & {e.episode_index for e in va})
    assert len(tr) + len(va) == len(frames.episodes)
    tr2, va2 = split_episodes(frames.episodes, 0.3, seed=0)
    assert [e.episode_index for e in va] == [e.episode_index for e in va2]


def test_one_real_minibatch(frames, processor, cfg):
    tr, _ = split_episodes(frames.episodes, 0.0, 0)
    ds = LiberoWindowDataset(frames, tr, 8)
    a, s = ds.all_train_arrays()
    norm = ActionStateNormalizer.fit(a, s)
    from monet_action.prompt import PromptBuilder

    pb = PromptBuilder(processor, cfg.prompt, "sft", 10)
    batch = Collator(pb, norm)([ds[0], ds[len(ds) - 1]])
    mi = batch["monet_inputs"]
    assert mi.input_ids.shape[0] == 2 and mi.num_latents == 10
    assert batch["actions"].shape == (2, 8, 7) and batch["state"].shape == (2, 8)
    assert batch["loss_mask"][1].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert torch.isfinite(batch["actions"]).all() and batch["actions"][..., :6].abs().max() <= 1
    assert mi.pixel_values.shape[0] == int(mi.image_grid_thw.prod(-1).sum())
    assert mi.image_grid_thw.shape[0] == 4  # 2 samples x 2 views
