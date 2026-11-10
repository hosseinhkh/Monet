"""Behaviour-cloning training for D0 / D1 / D2 / D4 / D5 with deterministic resume.

    python -m monet_action.train --config configs/d1_k10.yaml [--override train.max_steps=100 ...]

Re-running the same command resumes from ``<output_dir>/last`` (model, optimiser, RNG,
data-sampler position), so an interrupted run continues exactly where it stopped.
Artifacts in ``output_dir``: config.json, normalizer.json, split.json, param_report.json,
gradient_audit.json, train_log.jsonl, val_log.jsonl, memory_profile.json, ckpt_XXXXXXX/.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .config import ExperimentConfig, config_from_dict, load_config
from .libero_dataset import Collator, build_datasets
from .logging_utils import RunMonitor, get_logger
from .metrics import offline_metrics
from .normalization import ActionStateNormalizer
from .policy import MonetActionPolicy, build_policy
from .profiling import peak_memory_gb, reset_peak_memory
from .utils import append_jsonl, environment_report, rng_state, seed_everything, set_rng_state, write_json


# --------------------------------------------------------------------------- sampler
class ResumableSampler(Sampler[int]):
    """Infinite epoch-seeded permutations; ``start`` skips already-consumed samples."""

    def __init__(self, n: int, seed: int, start: int = 0):
        self.n, self.seed, self.start = n, seed, start

    def __iter__(self) -> Iterator[int]:
        epoch, offset = divmod(self.start, self.n)
        while True:
            g = torch.Generator().manual_seed(self.seed * 100003 + epoch)
            perm = torch.randperm(self.n, generator=g).tolist()
            for i in perm[offset:]:
                yield i
            epoch, offset = epoch + 1, 0

    def __len__(self) -> int:  # pragma: no cover
        return 2 ** 62


def lr_lambda(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / max(1, cfg.warmup_steps)
    prog = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def param_groups(policy: MonetActionPolicy, tcfg) -> List[Dict]:
    groups = {"lora": [], "state": [], "head": [], "head_nodecay": []}
    for n, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            groups["lora"].append(p)
        elif n.startswith("state_projector"):
            groups["state"].append(p)
        elif p.ndim < 2 or "embedding" in n or "null_token" in n or "token_pos" in n:
            groups["head_nodecay"].append(p)
        else:
            groups["head"].append(p)
    out = []
    if groups["lora"]:
        out.append({"name": "lora", "params": groups["lora"], "lr": tcfg.lr_lora, "weight_decay": 0.0})
    if groups["state"]:
        out.append({"name": "state", "params": groups["state"], "lr": tcfg.lr_state, "weight_decay": tcfg.weight_decay})
    out.append({"name": "head", "params": groups["head"], "lr": tcfg.lr_head, "weight_decay": tcfg.weight_decay})
    out.append({"name": "head_nodecay", "params": groups["head_nodecay"], "lr": tcfg.lr_head, "weight_decay": 0.0})
    return out


GRAD_GROUPS = {
    "action_head": lambda n: n.startswith("action_head"),
    "conditioner": lambda n: n.startswith("conditioner"),
    "conditioner_latent_proj": lambda n: n.startswith("conditioner.lat_proj"),
    "state_projector": lambda n: n.startswith("state_projector"),
    "monet_lora": lambda n: "lora_" in n,
    "learned_queries": lambda n: n.startswith("reasoner.learned_queries"),
}


def grad_report(policy) -> Dict[str, Dict[str, float]]:
    rep = {}
    for g, pred in GRAD_GROUPS.items():
        norms, finite, n_params, n_nonzero = [], True, 0, 0
        for n, p in policy.named_parameters():
            if p.requires_grad and pred(n):
                n_params += 1
                if p.grad is not None:
                    finite &= bool(torch.isfinite(p.grad).all())
                    gn = float(p.grad.float().norm())
                    norms.append(gn)
                    n_nonzero += int(gn > 0)
        if n_params:
            rep[g] = {"grad_norm": float(np.sqrt(np.sum(np.square(norms)))) if norms else 0.0, "finite": finite,
                      "n_tensors": n_params, "n_tensors_nonzero_grad": n_nonzero}
    return rep


def assert_gradients(rep: Dict, cfg: ExperimentConfig) -> None:
    must = ["action_head", "conditioner"]
    if cfg.state.inject_into_monet:
        must.append("state_projector")
    if cfg.monet.lora_enabled:
        must.append("monet_lora")
    if cfg.monet.reasoning_mode == "learned_query" and cfg.monet.num_latents > 0 and cfg.conditioner.feature_source != "vl_only":
        must.append("learned_queries")
    if cfg.monet.num_latents > 0 and cfg.conditioner.feature_source != "vl_only":
        must.append("conditioner_latent_proj")
    for g in must:
        r = rep.get(g)
        if r is None or not r["finite"] or r["grad_norm"] <= 0.0:
            raise AssertionError(f"gradient audit failed for '{g}': {r}")


# --------------------------------------------------------------------------- checkpoint io
def save_checkpoint(out: Path, step: int, policy, opt, consumed: int, cfg, keep_last: int,
                    keep_every: int = 0) -> Path:
    d = out / f"ckpt_{step:07d}"
    tmp = out / f".tmp_ckpt_{step:07d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    torch.save(policy.trainable_state_dict(), tmp / "trainable.pt")
    torch.save(opt.state_dict(), tmp / "optimizer.pt")
    torch.save(rng_state(), tmp / "rng.pt")
    write_json(tmp / "meta.json", {"step": step, "consumed_samples": consumed, "config": cfg.to_dict()})
    if d.exists():
        shutil.rmtree(d)
    tmp.rename(d)
    (out / "last").write_text(d.name)
    ckpts = sorted(out.glob("ckpt_*"))
    for old in ckpts[:-keep_last] if keep_last > 0 else []:
        if keep_every > 0 and int(old.name.split("_")[1]) % keep_every == 0:
            continue
        shutil.rmtree(old, ignore_errors=True)
    return d


def latest_checkpoint(out: Path) -> Optional[Path]:
    f = out / "last"
    if f.exists() and (out / f.read_text().strip()).exists():
        return out / f.read_text().strip()
    return None


def load_policy_from_checkpoint(ckpt_dir: str, device="cpu", overrides: Optional[List[str]] = None,
                                tokenizer_path=None):
    """Rebuild the policy from ``ckpt_dir/meta.json`` + trainable weights + run normalizer."""
    ckpt = Path(ckpt_dir)
    if (ckpt / "last").exists():
        ckpt = ckpt / (ckpt / "last").read_text().strip()
    meta = json.loads((ckpt / "meta.json").read_text())
    cfg = config_from_dict(meta["config"])
    if overrides:
        from .config import _deep_update, parse_overrides

        cfg = config_from_dict(_deep_update(cfg.to_dict(), parse_overrides(overrides)))
    norm = ActionStateNormalizer.load(ckpt.parent / "normalizer.json")
    policy, _ = build_policy(cfg, device=device, normalizer=norm, tokenizer_path=tokenizer_path)
    policy.load_trainable_state_dict(torch.load(ckpt / "trainable.pt", map_location="cpu"))
    policy.eval()
    return policy, cfg, meta


# --------------------------------------------------------------------------- main loop
def representation_report(policy: MonetActionPolicy, cfg: ExperimentConfig, example_instruction: str) -> Dict:
    """What exactly conditions the action head (proposal 2.1: log hidden layers and pooling)."""
    pb = policy.prompt_builder
    ids = pb.token_ids(example_instruction, len(cfg.prompt.camera_keys))
    m = cfg.monet
    return {
        "latent_layer": m.latent_layer, "vl_layer": m.vl_layer,
        "layer_note": "-1 == final post-RMSNorm hidden state (the vector Monet feeds back)",
        "pooling": "none: token-level conditioning (every valid prompt token + each latent step)",
        "num_latents_K": m.num_latents, "latent_convention": m.latent_convention,
        "recurrence_impl": m.recurrence_impl, "feedback_grad": m.feedback_grad, "reasoning_mode": m.reasoning_mode,
        "feature_source": cfg.conditioner.feature_source, "vl_token_select": cfg.conditioner.vl_token_select,
        "vl_max_tokens": cfg.conditioner.vl_max_tokens,
        "token_layout": "[NULL] + [VL_0..VL_{P-1}] + [LAT_0..LAT_{K-1}] (type + latent-order embeddings)",
        "state_tokens_in_prompt": cfg.prompt.num_state_tokens if cfg.state.inject_into_monet else 0,
        "state_to_action_head": cfg.state.state_to_action_head,
        "tool_result_prev_actions": cfg.tool_result.num_prev_actions,
        "tool_result_tokens": pb.n_tool,
        "image_tokens_per_view": pb.tokens_per_image, "camera_keys": cfg.prompt.camera_keys,
        "prompt_tokens_incl_latent_slots": len(ids), "slot_start_unpadded": len(ids) - m.num_latents
        - (1 if cfg.prompt.append_latent_end_token else 0),
        "example_prompt": pb.prompt_text(example_instruction, len(cfg.prompt.camera_keys)).replace(
            "<|image_pad|>" * pb.tokens_per_image, f"<|image_pad|>x{pb.tokens_per_image}"),
    }


def _sim_eval(policy, cfg: ExperimentConfig, out: Path, step: int, mon) -> None:
    """Optional closed-loop check on development init states; never interrupts training."""
    import copy

    from .evaluate import closed_loop

    ec = copy.deepcopy(cfg.eval)
    ec.task_ids, ec.seeds = list(cfg.train.sim_eval_tasks), [0]
    ec.episodes_per_task, ec.init_state_offset = cfg.train.sim_eval_episodes, cfg.train.sim_eval_init_offset
    ec.init_state_policy, ec.save_videos, ec.intervention = "fixed", 1, None
    ec.num_parallel_envs = min(ec.num_parallel_envs, cfg.train.sim_eval_episodes)
    policy.eval()
    t0 = time.time()
    try:
        s = closed_loop(policy, cfg, ec, out / "sim_eval" / f"step_{step:07d}", tag=f"@{step}", monitor=False)
        rec = {"step": step, "success_rate": s["success_rate"], "n_episodes": s["n_episodes"],
               "per_task": {str(t): {"successes": v["successes"], "trials": v["trials"]} for t, v in s["per_task"].items()},
               "wall_s": time.time() - t0, "time": time.time()}
        append_jsonl(out / "sim_eval_log.jsonl", rec)
        mon.event("sim_eval", step=step, success_rate=s["success_rate"], n_episodes=s["n_episodes"])
        mon.logger.info("sim eval @%d: success %.3f over %d episodes (%.0fs)", step, s["success_rate"],
                        s["n_episodes"], time.time() - t0)
    except Exception as e:  # e.g. no EGL on this node: report and keep training
        import traceback

        mon.event("sim_eval_failed", step=step, error=repr(e))
        mon.logger.warning("sim eval @%d failed (training continues):\n%s", step, traceback.format_exc())
    finally:
        policy.train()


def train(cfg: ExperimentConfig, device: str = "cuda", tokenizer_path: Optional[str] = None,
          max_minutes: Optional[float] = None) -> Dict:
    out = Path(cfg.train.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mon = RunMonitor(out, kind="train", name=cfg.name, variant=cfg.variant, max_steps=cfg.train.max_steps,
                     system_interval_s=cfg.train.system_monitor_interval_s,
                     extra={"device": device, "num_latents": cfg.monet.num_latents, "lora": cfg.monet.lora_enabled})
    with mon:
        return _train(cfg, out, mon, device, tokenizer_path, max_minutes)


def _train(cfg: ExperimentConfig, out: Path, mon: RunMonitor, device: str, tokenizer_path: Optional[str],
           max_minutes: Optional[float]) -> Dict:
    log = mon.logger
    seed_everything(cfg.train.seed, cfg.train.deterministic)
    write_json(out / "config.json", cfg.to_dict())
    write_json(out / "environment.json", environment_report())
    log.info("run %s (%s) -> %s on %s", cfg.name, cfg.variant, out, device)

    train_ds, val_ds, normalizer, split = build_datasets(cfg.data, cfg.action_head.action_horizon,
                                                         cfg.tool_result.num_prev_actions)
    log.info("data: %d train windows (%d episodes), %d val windows (%d episodes)", len(train_ds),
             len(train_ds.episodes), len(val_ds), len(val_ds.episodes))
    resume = latest_checkpoint(out)
    if resume is not None and (out / "normalizer.json").exists():
        normalizer = ActionStateNormalizer.load(out / "normalizer.json")  # keep the original stats
    else:
        normalizer.save(out / "normalizer.json")
        write_json(out / "split.json", split)

    policy, load_info = build_policy(cfg, device=device, normalizer=normalizer, tokenizer_path=tokenizer_path)
    if cfg.train.init_from:
        ck = Path(cfg.train.init_from)
        if (ck / "last").exists():
            ck = ck / (ck / "last").read_text().strip()
        sd = torch.load(ck / "trainable.pt", map_location="cpu")
        policy.load_trainable_state_dict(sd, strict=False)
        append_jsonl(out / "train_log.jsonl", {"event": "init_from", "path": str(ck), "n_tensors": len(sd), "time": time.time()})
        mon.event("init_from", path=str(ck), n_tensors=len(sd))
        log.info("initialised %d trainable tensors from %s", len(sd), ck)
    report = policy.parameter_report()
    write_json(out / "param_report.json", {**report, "monet_load_info": load_info})
    log.info("parameters: %s", json.dumps(report))
    rep = representation_report(policy, cfg, train_ds.episodes[0].instruction if train_ds.episodes else "task")
    write_json(out / "representation.json", rep)
    log.info("representation: K=%d layer=%d pooling=token-level feature_source=%s convention=%s",
             cfg.monet.num_latents, cfg.monet.latent_layer, cfg.conditioner.feature_source, cfg.monet.latent_convention)

    opt = torch.optim.AdamW(param_groups(policy, cfg.train), betas=(0.9, 0.95))
    base_lrs = [g["lr"] for g in opt.param_groups]
    start_step, consumed = 0, 0
    if resume is not None:
        meta = json.loads((resume / "meta.json").read_text())
        policy.load_trainable_state_dict(torch.load(resume / "trainable.pt", map_location="cpu"))
        opt.load_state_dict(torch.load(resume / "optimizer.pt", map_location="cpu"))
        set_rng_state(torch.load(resume / "rng.pt", weights_only=False))
        start_step, consumed = meta["step"], meta["consumed_samples"]
        append_jsonl(out / "train_log.jsonl", {"event": "resume", "from": str(resume), "step": start_step, "time": time.time()})
        mon.event("resume", path=str(resume), step=start_step)
        log.info("resumed from %s at step %d", resume, start_step)
    mon.update(step=start_step, samples=consumed)

    collate = Collator(policy.prompt_builder, normalizer, augment=cfg.data.augment, seed=cfg.train.seed)
    micro = cfg.train.batch_size
    loader = DataLoader(train_ds, batch_size=micro, sampler=ResumableSampler(len(train_ds), cfg.train.seed, consumed),
                        collate_fn=collate, num_workers=cfg.data.num_workers, drop_last=True,
                        persistent_workers=cfg.data.num_workers > 0, pin_memory=device.startswith("cuda"))
    val_collate = Collator(policy.prompt_builder, normalizer, augment=False)
    val_loader = DataLoader(val_ds, batch_size=micro, shuffle=False, collate_fn=val_collate,
                            num_workers=cfg.data.num_workers, drop_last=False) if len(val_ds) else None
    overfit_loader = None
    if cfg.train.overfit_eval_every > 0:
        overfit_loader = DataLoader(train_ds, batch_size=micro, shuffle=False, collate_fn=val_collate, num_workers=0)
    tb = _tensorboard(out) if cfg.train.tensorboard else None
    group_names = [g.get("name", str(i)) for i, g in enumerate(opt.param_groups)]
    it = iter(loader)
    policy.train()
    reset_peak_memory()
    t_start = time.time()
    step = start_step
    last_metrics: Dict = {}
    recent_dt: List[float] = []
    while step < cfg.train.max_steps:
        t0 = time.time()
        scale = lr_lambda(step, cfg.train)
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * scale
        opt.zero_grad(set_to_none=True)
        agg: Dict[str, float] = {}
        t_data = 0.0
        for _ in range(cfg.train.grad_accum):
            td = time.time()
            batch = next(it)
            t_data += time.time() - td
            loss, m = policy.compute_loss(batch)
            (loss / cfg.train.grad_accum).backward()
            consumed += micro
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v / cfg.train.grad_accum
        grep = None
        if step - start_step < cfg.train.gradient_audit_steps or step % cfg.train.log_every == 0:
            grep = grad_report(policy)
        if step - start_step < cfg.train.gradient_audit_steps:
            assert_gradients(grep, cfg)
            write_json(out / "gradient_audit.json", {"step": step, "report": grep, "passed": True})
            mon.event("gradient_audit_passed", step=step)
        gnorm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], cfg.train.grad_clip)
        if not torch.isfinite(gnorm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        opt.step()
        step += 1
        dt = time.time() - t0
        recent_dt = (recent_dt + [dt])[-50:]
        if step == start_step + 3:
            write_json(out / "memory_profile.json", {"peak_gb_after_3_steps": peak_memory_gb(),
                                                     "micro_batch": micro, "grad_accum": cfg.train.grad_accum})
        if step % cfg.train.log_every == 0 or step == start_step + 1:
            eta = float(np.mean(recent_dt)) * (cfg.train.max_steps - step)
            last_metrics = {"step": step, **agg, "grad_norm": float(gnorm), "lr_scale": scale,
                            "lrs": {n: g["lr"] for n, g in zip(group_names, opt.param_groups)},
                            "step_time_s": dt, "data_time_s": t_data, "peak_mem_gb": peak_memory_gb(),
                            "samples": consumed, "time": time.time(), "grad_groups": grep}
            append_jsonl(out / "train_log.jsonl", last_metrics)
            mon.update(step=step, samples=consumed, last_loss=agg.get("loss"), eta_s=eta,
                       step_time_s=float(np.mean(recent_dt)), peak_mem_gb=peak_memory_gb())
            log.info("step %d/%d loss %.4f gnorm %.3f lr x%.3f %.2fs/step (data %.2fs) mem %.1fGB eta %.1fh",
                     step, cfg.train.max_steps, agg.get("loss", 0.0), float(gnorm), scale, dt, t_data,
                     peak_memory_gb(), eta / 3600)
            if tb is not None:
                for k, v in agg.items():
                    tb.add_scalar(f"train/{k}", v, step)
                tb.add_scalar("train/grad_norm", float(gnorm), step)
                tb.add_scalar("train/lr_scale", scale, step)
        if val_loader is not None and (step % cfg.train.val_every == 0 or step == cfg.train.max_steps):
            policy.eval()
            vm = offline_metrics(policy, val_loader, max_batches=cfg.train.val_batches,
                                 num_steps=cfg.action_head.num_inference_steps)
            policy.train()
            append_jsonl(out / "val_log.jsonl", {"step": step, **vm, "time": time.time()})
            mon.event("val", step=step, **{k: v for k, v in vm.items() if k != "val_batches"})
            log.info("val @%d: %s", step, json.dumps({k: round(v, 5) for k, v in vm.items()}))
            if tb is not None:
                for k, v in vm.items():
                    tb.add_scalar(f"val/{k}", v, step)
        if overfit_loader is not None and (step % cfg.train.overfit_eval_every == 0 or step == start_step + 1
                                           or step == cfg.train.max_steps):
            policy.eval()
            om = offline_metrics(policy, overfit_loader, max_batches=1000, num_steps=cfg.action_head.num_inference_steps)
            policy.train()
            append_jsonl(out / "overfit_log.jsonl", {"step": step, **om, "time": time.time()})
            log.info("overfit-eval @%d: fm_loss %.4f action_mse %.4f", step, om["fm_loss"], om["action_mse"])
        if cfg.train.sim_eval_every > 0 and step % cfg.train.sim_eval_every == 0:
            _sim_eval(policy, cfg, out, step, mon)
        if step % cfg.train.save_every == 0 or step == cfg.train.max_steps:
            d = save_checkpoint(out, step, policy, opt, consumed, cfg, cfg.train.keep_last, cfg.train.keep_every)
            mon.event("checkpoint", step=step, path=str(d))
            log.info("checkpoint %s", d)
        if max_minutes is not None and (time.time() - t_start) / 60 > max_minutes:
            d = save_checkpoint(out, step, policy, opt, consumed, cfg, cfg.train.keep_last, cfg.train.keep_every)
            mon.event("checkpoint", step=step, path=str(d))
            mon.event("time_limit", step=step, max_minutes=max_minutes)
            log.info("stopping after %.1f min at step %d (resumable)", max_minutes, step)
            break
    if tb is not None:
        tb.close()
    mon.update(step=step, samples=consumed)
    return {"step": step, "last": last_metrics, "output_dir": str(out)}


def _tensorboard(out: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(str(out / "tensorboard"))
    except Exception as e:  # pragma: no cover
        get_logger().warning("tensorboard requested but unavailable: %s", e)
        return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tokenizer-path", default=None)
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.override)
    res = train(cfg, a.device, a.tokenizer_path, a.max_minutes)
    print(json.dumps({k: v for k, v in res.items() if k != "last"}, indent=2, default=str))


if __name__ == "__main__":
    main()
