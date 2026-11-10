"""LIBERO-Spatial behaviour-cloning windows from the LeRobot v3 dataset.

Dataset: ``lerobot/libero_spatial_image`` @ d86c0b9 (Apache-2.0; 432 successful demos,
52,970 frames, 10 tasks, regenerated OpenVLA-style without no-op actions, 256x256 images).
One frame == one ``env.step`` of the 20 Hz OSC_POSE controller (verified in G0: per-step
eef displacement ~ 11-13 mm per unit action).

Sample at frame t of an episode of length T:
    images      agent-view + wrist RGB at t (HWC uint8, already upright == sim rot180 view)
    state       8D proprioception at t
    actions     a[t : t+H]  (H = action horizon), padded by repeating a[T-1]
    loss_mask   1 for t+h < T, 0 for padded steps (never crosses an episode boundary)
    prev_actions a[t-n : t] (TOOL_RESULT context: what was executed just before t), with
                prev_valid = 0 for indices before the episode start (zero-filled, never from
                the previous episode)
Splits are episode-disjoint and stratified by task; normalisation statistics are fit on the
training split only.
"""
from __future__ import annotations

import glob
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from .normalization import ActionStateNormalizer

CAMERA_COLUMNS = {"image": "observation.images.image", "wrist_image": "observation.images.wrist_image"}


@dataclass
class EpisodeInfo:
    episode_index: int
    task_index: int
    instruction: str
    start: int  # row offset into the in-memory frame table
    length: int


class LeRobotLiberoFrames:
    """In-memory frame table: numeric columns + PNG bytes in one flat buffer (fork/COW friendly)."""

    def __init__(self, root: str, file_limit: Optional[int] = None, task_indices: Optional[List[int]] = None,
                 max_episodes_per_task: Optional[int] = None, camera_keys=("image", "wrist_image")):
        self.root = Path(root)
        if not (self.root / "meta" / "info.json").exists():
            raise FileNotFoundError(f"{root} is not a LeRobot dataset (missing meta/info.json). "
                                    f"Run scripts/download_data.sh first.")
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        assert self.info["features"]["observation.state"]["shape"] == [8], "expected 8D state"
        assert self.info["features"]["action"]["shape"] == [7], "expected 7D action"
        tasks = pq.read_table(self.root / "meta" / "tasks.parquet").to_pandas()
        self.task_text = {int(r.task_index): str(t) for t, r in tasks.iterrows()}
        self.camera_keys = list(camera_keys)

        files = sorted(glob.glob(str(self.root / "data" / "chunk-*" / "file-*.parquet")))
        if not files:
            raise FileNotFoundError(f"no parquet files under {self.root}/data")
        if file_limit:
            files = files[:file_limit]
        cols = ["episode_index", "frame_index", "task_index", "observation.state", "action"]
        cols += [CAMERA_COLUMNS[k] for k in self.camera_keys]
        ep_l, fr_l, tk_l, st_l, ac_l = [], [], [], [], []
        img_bytes: Dict[str, List[bytes]] = {k: [] for k in self.camera_keys}
        for f in files:
            t = pq.read_table(f, columns=cols)
            ep_l.append(t.column("episode_index").to_numpy())
            fr_l.append(t.column("frame_index").to_numpy())
            tk_l.append(t.column("task_index").to_numpy())
            st_l.append(np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32))
            ac_l.append(np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float32))
            for k in self.camera_keys:
                img_bytes[k].extend(x["bytes"] for x in t.column(CAMERA_COLUMNS[k]).to_pylist())
        ep, fr, tk = np.concatenate(ep_l), np.concatenate(fr_l), np.concatenate(tk_l)
        st, ac = np.concatenate(st_l), np.concatenate(ac_l)
        order = np.lexsort((fr, ep))  # sort by (episode, frame)
        self.episode_index, self.frame_index, self.task_index = ep[order], fr[order], tk[order]
        self.state, self.action = st[order], ac[order]
        self._img_buf, self._img_off = {}, {}
        for k in self.camera_keys:
            blobs = [img_bytes[k][i] for i in order]
            lens = np.fromiter((len(b) for b in blobs), dtype=np.int64, count=len(blobs))
            off = np.zeros(len(blobs) + 1, dtype=np.int64)
            np.cumsum(lens, out=off[1:])
            self._img_buf[k] = np.frombuffer(b"".join(blobs), dtype=np.uint8)
            self._img_off[k] = off

        # Episodes (complete only: a partially downloaded episode would silently be truncated)
        episodes: List[EpisodeInfo] = []
        bounds = np.flatnonzero(np.r_[True, self.episode_index[1:] != self.episode_index[:-1], True])
        ep_len_meta = self._episode_lengths_from_meta()
        per_task_count: Dict[int, int] = {}
        for s, e in zip(bounds[:-1], bounds[1:]):
            ei, ti = int(self.episode_index[s]), int(self.task_index[s])
            frames = self.frame_index[s:e]
            if not np.array_equal(frames, np.arange(len(frames))):
                raise RuntimeError(f"episode {ei} has non-contiguous frames")
            if ep_len_meta and ep_len_meta.get(ei, len(frames)) != len(frames):
                continue  # incomplete episode (split across a file we did not load)
            if task_indices is not None and ti not in task_indices:
                continue
            if max_episodes_per_task is not None and per_task_count.get(ti, 0) >= max_episodes_per_task:
                continue
            per_task_count[ti] = per_task_count.get(ti, 0) + 1
            episodes.append(EpisodeInfo(ei, ti, self.task_text[ti], int(s), int(e - s)))
        self.episodes = episodes

    def _episode_lengths_from_meta(self) -> Dict[int, int]:
        out = {}
        for f in glob.glob(str(self.root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True):
            t = pq.read_table(f, columns=["episode_index", "length"]).to_pandas()
            out.update({int(a): int(b) for a, b in zip(t.episode_index, t.length)})
        return out

    def image(self, key: str, row: int) -> np.ndarray:
        off = self._img_off[key]
        raw = self._img_buf[key][off[row]:off[row + 1]].tobytes()
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def split_episodes(episodes: List[EpisodeInfo], val_fraction: float, seed: int) -> Tuple[List[EpisodeInfo], List[EpisodeInfo]]:
    """Episode-disjoint split, stratified by task, deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    by_task: Dict[int, List[EpisodeInfo]] = {}
    for e in episodes:
        by_task.setdefault(e.task_index, []).append(e)
    train, val = [], []
    for t in sorted(by_task):
        eps = sorted(by_task[t], key=lambda e: e.episode_index)
        perm = rng.permutation(len(eps))
        n_val = int(math.ceil(val_fraction * len(eps))) if len(eps) > 1 and val_fraction > 0 else 0
        val_idx = set(perm[:n_val].tolist())
        for i, e in enumerate(eps):
            (val if i in val_idx else train).append(e)
    return train, val


class LiberoWindowDataset(Dataset):
    def __init__(self, frames: LeRobotLiberoFrames, episodes: List[EpisodeInfo], horizon: int,
                 max_windows: Optional[int] = None, num_prev_actions: int = 0):
        self.frames, self.episodes, self.H = frames, episodes, horizon
        self.n_prev = num_prev_actions
        idx = []
        for k, e in enumerate(episodes):
            for t in range(e.length):
                idx.append((k, t))
        if max_windows is not None:
            idx = idx[:max_windows]
        self.index = idx

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict:
        k, t = self.index[i]
        e = self.episodes[k]
        rows = e.start + np.minimum(np.arange(t, t + self.H), e.length - 1)  # clamp inside the episode
        mask = (np.arange(t, t + self.H) < e.length).astype(np.float32)
        row = e.start + t
        n = self.n_prev
        prev_idx = np.arange(t - n, t)
        prev_valid = (prev_idx >= 0).astype(np.float32)
        prev = self.frames.action[e.start + np.clip(prev_idx, 0, None)].copy() if n else np.zeros((0, 7), np.float32)
        prev[prev_valid == 0] = 0.0
        return {
            "prev_actions": prev,
            "prev_valid": prev_valid,
            "images": [self.frames.image(c, row) for c in self.frames.camera_keys],
            "state": self.frames.state[row].copy(),
            "actions": self.frames.action[rows].copy(),
            "loss_mask": mask,
            "instruction": e.instruction,
            "task_index": e.task_index,
            "episode_index": e.episode_index,
            "frame_index": t,
        }

    def all_train_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        rows = np.concatenate([np.arange(e.start, e.start + e.length) for e in self.episodes])
        return self.frames.action[rows], self.frames.state[rows]


def _augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild random-resized-crop (90% area) + brightness/contrast jitter. Off by default."""
    H, W, _ = img.shape
    s = math.sqrt(rng.uniform(0.9, 1.0))
    h, w = int(H * s), int(W * s)
    y, x = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1)
    out = Image.fromarray(img[y:y + h, x:x + w]).resize((W, H), Image.BILINEAR)
    a = np.asarray(out).astype(np.float32)
    a = (a - a.mean()) * rng.uniform(0.9, 1.1) + a.mean() + rng.uniform(-10, 10)
    return np.clip(a, 0, 255).astype(np.uint8)


class Collator:
    """Builds Monet prompts (tokenisation + Qwen image processing) and normalised tensors.

    Runs inside DataLoader workers so CPU preprocessing overlaps GPU compute.
    """

    def __init__(self, prompt_builder, normalizer: ActionStateNormalizer, augment: bool = False, seed: int = 0):
        self.pb, self.norm, self.augment = prompt_builder, normalizer, augment
        self.rng = np.random.default_rng(seed)

    def __call__(self, samples: List[Dict]) -> Dict:
        views = [s["images"] for s in samples]
        if self.augment:
            views = [[_augment(im, self.rng) for im in v] for v in views]
        inputs = self.pb.build(views, [s["instruction"] for s in samples])
        state = torch.from_numpy(np.stack([s["state"] for s in samples]))
        actions = torch.from_numpy(np.stack([s["actions"] for s in samples]))
        prev = torch.from_numpy(np.stack([s["prev_actions"] for s in samples]))
        prev_valid = torch.from_numpy(np.stack([s["prev_valid"] for s in samples]))
        if prev.shape[1] > 0:
            prev = self.norm.action.normalize(prev).float() * prev_valid.unsqueeze(-1)
        return {
            "monet_inputs": inputs,
            "prev_actions": prev.float(),
            "prev_valid": prev_valid.float(),
            "state": self.norm.state.normalize(state).float(),
            "actions": self.norm.action.normalize(actions).float(),
            "raw_actions": actions,
            "loss_mask": torch.from_numpy(np.stack([s["loss_mask"] for s in samples])),
            "task_index": torch.tensor([s["task_index"] for s in samples]),
            "episode_index": torch.tensor([s["episode_index"] for s in samples]),
            "frame_index": torch.tensor([s["frame_index"] for s in samples]),
        }


def build_datasets(data_cfg, horizon: int, num_prev_actions: int = 0):
    frames = LeRobotLiberoFrames(data_cfg.root, file_limit=data_cfg.file_limit, task_indices=data_cfg.task_indices,
                                 max_episodes_per_task=data_cfg.max_episodes_per_task)
    train_eps, val_eps = split_episodes(frames.episodes, data_cfg.val_fraction, data_cfg.split_seed)
    train = LiberoWindowDataset(frames, train_eps, horizon, max_windows=data_cfg.max_windows,
                                num_prev_actions=num_prev_actions)
    val = LiberoWindowDataset(frames, val_eps, horizon, num_prev_actions=num_prev_actions)
    a, s = LiberoWindowDataset(frames, train_eps, horizon).all_train_arrays()
    normalizer = ActionStateNormalizer.fit(a, s, data_cfg.action_norm, data_cfg.state_norm)
    split = {"train_episodes": [e.episode_index for e in train_eps], "val_episodes": [e.episode_index for e in val_eps]}
    return train, val, normalizer, split
