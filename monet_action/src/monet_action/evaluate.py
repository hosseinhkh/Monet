"""Evaluation: closed-loop LIBERO, open-loop held-out metrics, latency profiling, comparison.

  # closed loop (all 10 LIBERO-Spatial tasks, 3 seeds x 10 episodes, E=1, 4 Euler steps)
  python -m monet_action.evaluate closed_loop --checkpoint runs/d1_k10 --config configs/eval.yaml
  # D3 latent interventions on the D2 checkpoint
  python -m monet_action.evaluate closed_loop --checkpoint runs/d2_lora --intervention shuffle_order
  # solver sensitivity / open-loop
  python -m monet_action.evaluate offline --checkpoint runs/d2_lora --steps 2 4 8
  # batch-1 latency profile (no simulator)
  python -m monet_action.evaluate latency --checkpoint runs/d1_k10 --steps 2 4 8
  # comparison table + paired statistics
  python -m monet_action.evaluate compare results/eval/D0 results/eval/D1 results/eval/D2

Protocol: initial states are the fixed LIBERO benchmark states (matched across variants);
the evaluation seed sets the DiT sampling noise and, with init_state_policy=seed_offset, which
block of benchmark init states is used. Success is LIBERO's own ``done`` flag, used only for
scoring. Results are paused-simulator task success, not evidence of real-time feasibility.
"""
from __future__ import annotations

import argparse

from monet_action.runtime_env import apply_defaults  # noqa: E402  (before torch: GPU pinning, MUJOCO_GL)

apply_defaults()
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import EvalConfig
from .logging_utils import RunMonitor, get_logger
from .metrics import (
    aggregate_episodes,
    bootstrap_diff_ci,
    markdown_table,
    mcnemar_exact,
    offline_metrics,
    with_derived_latency,
)
from .profiling import peak_memory_gb, percentiles, reset_peak_memory
from .tool_runtime import ActionTool, run_episodes_lockstep
from .utils import append_jsonl, environment_report, write_json


def _eval_cfg(cfg, args) -> EvalConfig:
    ec = cfg.eval
    if args.eval_config:
        from .config import load_config

        ec = load_config(args.eval_config).eval
    for k in ("seeds", "task_ids", "episodes_per_task", "execution_prefix", "num_inference_steps",
              "num_parallel_envs", "intervention", "output_dir", "max_steps", "save_videos", "init_state_offset"):
        v = getattr(args, k, None)
        if v is not None:
            setattr(ec, k, v)
    return ec


def _save_video(frames, path: Path, fps: int = 20) -> None:
    try:
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(path), frames, fps=fps)
    except Exception as e:  # pragma: no cover
        print(f"video save failed: {e}")


def warm_latent_bank(policy, ec: EvalConfig, task_ids: List[int]) -> None:
    """Fill the swap-intervention bank with genuine latents from each task's first observation."""
    from .libero_env import LiberoEnv

    for t in task_ids:
        env = LiberoEnv(ec.suite, t, ec.camera_size, 0, jpeg_quality=ec.jpeg_quality)
        obs = env.reset(len(env.init_states) - 1, ec.num_wait_steps)  # dev block (49), never a final-eval state
        pred = policy.predict([obs], num_steps=1)
        policy.latent_bank.add(pred.latents, [t])
        env.close()
    if len(set(task_ids)) < 2:
        get_logger("monet_action.eval").warning(
            "swap intervention needs >= 2 tasks; with %d task(s) latents are left unchanged", len(set(task_ids)))


def init_state_indices(ec: EvalConfig, seed_index: int) -> List[int]:
    """Benchmark init states for one (seed, task): matched across variants. Final reporting uses
    0..29 (3 seeds x 10 episodes, ``seed_offset``); development rollouts set init_state_offset>=30."""
    base = ec.init_state_offset + (seed_index * ec.episodes_per_task if ec.init_state_policy == "seed_offset" else 0)
    return [base + e for e in range(ec.episodes_per_task)]


def closed_loop(policy, cfg, ec: EvalConfig, out_dir: Path, tag: str = "", monitor: bool = True) -> Dict:
    """Closed-loop LIBERO evaluation. With ``monitor`` the eval dir gets status.json, events.jsonl,
    system.jsonl and logs/eval.log (dashboard contract); resumable via episodes.jsonl."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not monitor:
        return _closed_loop(policy, cfg, ec, out_dir, tag, None)
    total = len(ec.seeds) * (len(ec.task_ids) if ec.task_ids is not None else 10) * ec.episodes_per_task
    mon = RunMonitor(out_dir, kind="eval", name=out_dir.name, variant=cfg.variant,
                     extra={"mode": "closed_loop", "episodes_total": total, "episodes_done": 0,
                            "intervention": ec.intervention, "execution_prefix": ec.execution_prefix})
    with mon:
        return _closed_loop(policy, cfg, ec, out_dir, tag, mon)


def _closed_loop(policy, cfg, ec: EvalConfig, out_dir: Path, tag: str, mon: Optional[RunMonitor]) -> Dict:
    from .libero_env import LiberoEnv, libero_task_table

    log = mon.logger if mon is not None else get_logger("monet_action.eval")
    table = libero_task_table(ec.suite)
    task_ids = ec.task_ids if ec.task_ids is not None else [t for t, _, _ in table]
    steps = ec.num_inference_steps or cfg.action_head.num_inference_steps
    write_json(out_dir / "eval_config.json", {"eval": ec.__dict__, "num_inference_steps": steps, "tag": tag,
                                              "model_config": cfg.to_dict(), "environment": environment_report(),
                                              "task_table": table})
    if ec.intervention == "swap":
        warm_latent_bank(policy, ec, task_ids)
    ep_file = out_dir / "episodes.jsonl"
    done_keys = set()
    if ep_file.exists():  # resumable evaluation
        for line in ep_file.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done_keys.add((r["seed"], r["task_id"], r["init_state_idx"]))
    n_done = len(done_keys)
    if mon is not None:
        mon.update(episodes_done=n_done)
    reset_peak_memory()
    # Environments are built once per task and reused for every seed: resets are seeded per
    # (task, init state), so the construction seed does not influence episodes.
    for t in task_ids:
        todo = {si: [i for i in init_state_indices(ec, si) if (seed, t, i) not in done_keys]
                for si, seed in enumerate(ec.seeds)}
        if not any(todo.values()):
            continue
        P = max(1, min(ec.num_parallel_envs, max(len(v) for v in todo.values())))
        envs = [LiberoEnv(ec.suite, t, ec.camera_size, 0, jpeg_quality=ec.jpeg_quality) for _ in range(P)]
        try:
            for si, seed in enumerate(ec.seeds):
                idxs = todo[si]
                for c0 in range(0, len(idxs), P):
                    chunk = idxs[c0:c0 + P]
                    g = torch.Generator().manual_seed(  # CPU generator: noise is drawn on CPU then moved
                        seed * 1_000_003 + t * 1009 + c0)
                    tool = ActionTool(policy, cfg.action_head.action_horizon, generator=g, intervention=ec.intervention)
                    rec_frames = [(c0 + j) < ec.save_videos for j in range(len(chunk))]
                    t0 = time.time()
                    recs = run_episodes_lockstep(envs[:len(chunk)], chunk, tool, max_steps=ec.max_steps,
                                                 execution_prefix=ec.execution_prefix, num_integration_steps=steps,
                                                 num_wait_steps=ec.num_wait_steps, record_frames=rec_frames,
                                                 task_ids=[t] * len(chunk), seed=seed)
                    for r in recs:
                        row = {k: v for k, v in r.__dict__.items() if k not in ("frames", "trace")}
                        row["wall_s"] = (time.time() - t0) / len(chunk)
                        append_jsonl(ep_file, row)
                        append_jsonl(out_dir / "traces" / f"seed{seed}_task{t}_init{r.init_state_idx}.jsonl",
                                     {"episode": row, "trace": r.trace})
                        if r.frames:
                            status = "success" if r.success else "failure"
                            _save_video(r.frames, out_dir / "videos" / f"seed{seed}_task{t}_init{r.init_state_idx}_{status}.mp4")
                    n_done += len(recs)
                    k = sum(r.success for r in recs)
                    log.info("[eval%s] seed %d task %d init %s: %d/%d success (%.0fs)", tag, seed, t, chunk, k,
                             len(recs), time.time() - t0)
                    if mon is not None:
                        mon.update(episodes_done=n_done, step=n_done)
                        mon.event("episodes", seed=seed, task_id=t, init_states=chunk, successes=k, n=len(recs))
        finally:
            for e in envs:
                e.close()
    records = [json.loads(line) for line in ep_file.read_text().splitlines() if line.strip()]
    summary = aggregate_episodes(records, ec.gpu_hourly_cost_usd)
    summary.update({"num_inference_steps": steps, "execution_prefix": ec.execution_prefix,
                    "intervention": ec.intervention, "gpu_peak_mem_gb": peak_memory_gb(),
                    "task_names": {t: n for t, n, _ in table}, "tag": tag,
                    "failure_tag_counts": _tag_counts(records),
                    "note": "paused-simulator task success; not a real-time or physical-safety claim"})
    write_json(out_dir / "summary.json", summary)
    if mon is not None:
        mon.update(success_rate=summary["success_rate"])
    return summary


def _tag_counts(records: List[Dict]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for r in records:
        for t in r.get("failure_tags", []) or []:
            c[t] = c.get(t, 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


@torch.no_grad()
def latency_profile(policy, cfg, steps_list: List[int], n_calls: int = 50, batch_size: int = 1) -> Dict:
    """Batch-1 observation-to-action latency split into preprocess / monet / conditioner / head."""
    from .libero_dataset import LeRobotLiberoFrames
    from .libero_env import PolicyObservation

    frames = LeRobotLiberoFrames(cfg.data.root, file_limit=1)
    ep = frames.episodes[0]
    obs = [PolicyObservation({k: frames.image(k, ep.start + i) for k in frames.camera_keys},
                             frames.state[ep.start + i], ep.instruction) for i in range(batch_size)]
    out = {}
    for steps in steps_list:
        for _ in range(3):
            policy.predict(obs, num_steps=steps)  # warm-up
        reset_peak_memory()
        comp: Dict[str, List[float]] = {}
        for _ in range(n_calls):
            p = policy.predict(obs, num_steps=steps)
            for k, v in with_derived_latency(p.timings_ms).items():
                comp.setdefault(k, []).append(v)
        out[f"steps_{steps}"] = {k: percentiles(v) for k, v in comp.items()} | {"peak_mem_gb": peak_memory_gb()}
    return {"K": cfg.monet.num_latents, "batch_size": batch_size, "device": str(policy.device), "profile": out}


def compare(dirs: List[str], out: Optional[str] = None) -> str:
    summaries, episodes = {}, {}
    for d in dirs:
        p = Path(d)
        s = json.loads((p / "summary.json").read_text())
        name = p.name if p.name not in summaries else f"{p.parent.name}/{p.name}"
        summaries[name] = s
        episodes[name] = {(r["seed"], r["task_id"], r["init_state_idx"]): int(r["success"])
                          for r in map(json.loads, (p / "episodes.jsonl").read_text().splitlines())}
    md = [markdown_table(summaries), "", "Paired comparisons (matched seed/task/init state):", "",
          "| A | B | n paired | rate A | rate B | diff B-A | 95% bootstrap CI | McNemar p |", "|---|---|---|---|---|---|---|---|"]
    names = list(summaries)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = episodes[names[i]], episodes[names[j]]
            keys = sorted(set(a) & set(b))
            if not keys:
                continue
            xa, xb = [a[k] for k in keys], [b[k] for k in keys]
            lo, hi = bootstrap_diff_ci(xa, xb)
            md.append(f"| {names[i]} | {names[j]} | {len(keys)} | {np.mean(xa):.3f} | {np.mean(xb):.3f} | "
                      f"{np.mean(xb) - np.mean(xa):+.3f} | [{lo:+.3f}, {hi:+.3f}] | {mcnemar_exact(xa, xb):.4f} |")
    text = "\n".join(md)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n")
    return text


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("closed_loop", "offline", "latency"):
        s = sub.add_parser(name)
        s.add_argument("--checkpoint", required=True)
        s.add_argument("--eval-config", default=None)
        s.add_argument("--override", nargs="*", default=[])
        s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        s.add_argument("--tokenizer-path", default=None)
        s.add_argument("--output-dir", dest="output_dir", default=None)
        s.add_argument("--tag", default="")
        if name == "closed_loop":
            s.add_argument("--seeds", type=int, nargs="*", default=None)
            s.add_argument("--task-ids", dest="task_ids", type=int, nargs="*", default=None)
            s.add_argument("--episodes-per-task", dest="episodes_per_task", type=int, default=None)
            s.add_argument("--execution-prefix", dest="execution_prefix", type=int, default=None)
            s.add_argument("--num-inference-steps", dest="num_inference_steps", type=int, default=None)
            s.add_argument("--num-parallel-envs", dest="num_parallel_envs", type=int, default=None)
            s.add_argument("--max-steps", dest="max_steps", type=int, default=None)
            s.add_argument("--save-videos", dest="save_videos", type=int, default=None)
            s.add_argument("--intervention", default=None)
            s.add_argument("--init-state-offset", dest="init_state_offset", type=int, default=None)
        else:
            s.add_argument("--steps", type=int, nargs="*", default=[2, 4, 8])
            s.add_argument("--max-batches", type=int, default=100)
            s.add_argument("--n-calls", type=int, default=50)
    c = sub.add_parser("compare")
    c.add_argument("dirs", nargs="+")
    c.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.mode == "compare":
        print(compare(a.dirs, a.out))
        return
    from .gpu import setup_gpu
    from .train import load_policy_from_checkpoint

    setup_gpu(a.device)  # before loading: GPU-holder hand-over, cap, MONET_GPU_RESERVE (scripts/common.sh pins GPU 0)
    policy, cfg, meta = load_policy_from_checkpoint(a.checkpoint, a.device, a.override, a.tokenizer_path)
    ec = _eval_cfg(cfg, a)
    default_name = f"{cfg.variant}_{cfg.name}" + (f"_{ec.intervention.replace(':', '')}" if getattr(ec, "intervention", None) else "")
    out_dir = Path(a.output_dir or Path(ec.output_dir) / default_name)
    if a.mode == "closed_loop":
        s = closed_loop(policy, cfg, ec, out_dir, a.tag)
        print(json.dumps({k: s[k] for k in ("success_rate", "success_ci95", "gpu_seconds_per_success")}, indent=2))
    elif a.mode == "offline":
        from torch.utils.data import DataLoader

        from .libero_dataset import Collator, LeRobotLiberoFrames, LiberoWindowDataset, split_episodes

        frames = LeRobotLiberoFrames(cfg.data.root, file_limit=cfg.data.file_limit, task_indices=cfg.data.task_indices,
                                     max_episodes_per_task=cfg.data.max_episodes_per_task)
        split = json.loads((Path(meta["config"]["train"]["output_dir"]) / "split.json").read_text()) \
            if (Path(meta["config"]["train"]["output_dir"]) / "split.json").exists() else None
        if split:
            val_eps = [e for e in frames.episodes if e.episode_index in set(split["val_episodes"])]
        else:
            _, val_eps = split_episodes(frames.episodes, cfg.data.val_fraction, cfg.data.split_seed)
        ds = LiberoWindowDataset(frames, val_eps, cfg.action_head.action_horizon,
                                 num_prev_actions=cfg.tool_result.num_prev_actions)
        dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                        collate_fn=Collator(policy.prompt_builder, policy.normalizer), num_workers=cfg.data.num_workers)
        if len(ds) == 0:
            print("WARNING: no held-out validation episodes for this run (val_fraction=0?)")
        res = {f"steps_{s}": offline_metrics(policy, dl, a.max_batches, s) for s in a.steps}
        write_json(out_dir / "offline.json", res)
        print(json.dumps(res, indent=2))
    elif a.mode == "latency":
        res = latency_profile(policy, cfg, a.steps, a.n_calls)
        write_json(out_dir / "latency.json", res)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
