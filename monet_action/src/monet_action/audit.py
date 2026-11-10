"""G0 audit: environment/source pins, genuine Monet latents, LIBERO data batch, simulator I/O.

  python -m monet_action.audit --config configs/audit.yaml --out results/audit [--skip-sim]

With ``monet.model_path=NOVAglow646/Monet-7B`` (GPU server) this is the Gate-0 evidence that
real Monet-7B latents are produced; with ``tiny-random`` it validates the same code paths on CPU.
Every check records numbers, not just pass/fail. Nothing here trains anything.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .config import load_config
from .utils import environment_report, write_json

PINNED_SOURCES = {
    "Monet (code, this repo's upstream)": ("https://github.com/NOVAglow646/Monet", "08939998d3d643a73a316e349faa34f420429153"),
    "Monet-7B checkpoint": ("https://huggingface.co/NOVAglow646/Monet-7B", "39c1e320a3e15c357bff911fda263923edbdda37"),
    "Isaac-GR00T (design reference only)": ("https://github.com/NVIDIA/Isaac-GR00T", "7aa0bca50f1097f8b17f5c274c98264b72a1419f"),
    "LIBERO": ("https://github.com/Lifelong-Robot-Learning/LIBERO", "8f1084e3132a39270c3a13ebe37270a43ece2a01"),
    "lerobot/libero_spatial_image dataset": ("https://huggingface.co/datasets/lerobot/libero_spatial_image", "d86c0b94922572b3b657e1d1a3d01f0952ddeb46"),
}


def _cos(a, b) -> float:
    return float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


def audit_monet(cfg, tokenizer_path=None, device="cpu", data_root=None) -> Dict:
    from .libero_dataset import LeRobotLiberoFrames
    from .monet_backbone import MonetLatentReasoner, authors_latent_mode_reference, vllm_runner_emulation
    from .monet_loader import load_monet, load_processor
    from .normalization import Stats
    from .prompt import PromptBuilder
    from .state_context import RobotStateProjector, embedding_rms

    rep: Dict = {"model_path": cfg.monet.model_path, "revision": cfg.monet.model_revision}
    t0 = time.time()
    model, info = load_monet(cfg.monet, device=device)
    model = model.to(device)
    from .gpu import setup_gpu

    rep_gpu = setup_gpu(device)
    rep["load_s"] = time.time() - t0
    rep["gpu"] = rep_gpu
    rep["load_info"] = {k: (v[:10] if isinstance(v, list) else v) for k, v in info.items()}
    rep["hidden_size"] = model.config.text_config.hidden_size
    rep["num_layers"] = model.config.text_config.num_hidden_layers
    rep["special_ids"] = {k: getattr(model.config, k) for k in ("latent_token_id", "latent_start_id", "latent_end_id", "answer_start_pattern")}
    proc = load_processor(cfg.monet, tokenizer_path)

    # real LIBERO observation(s)
    frames = LeRobotLiberoFrames(data_root or cfg.data.root, file_limit=1)
    eps = frames.episodes[:2]
    views = [[frames.image(k, e.start) for k in frames.camera_keys] for e in eps]
    instr = [e.instruction for e in eps]
    st = Stats.from_array(frames.state)
    state = torch.from_numpy((np.stack([frames.state[e.start] for e in eps]) - st.mean) / np.maximum(st.std, 1e-6)).float()

    D = rep["hidden_size"]
    torch.manual_seed(0)
    proj = RobotStateProjector(8, D, cfg.prompt.num_state_tokens, 256, init_rms=embedding_rms(model.model.language_model.embed_tokens)).to(device)
    dtype = model.model.language_model.embed_tokens.weight.dtype
    st_tok = proj(state.to(device)).to(dtype)
    K = 10
    res: Dict = {}
    lat_by_conv = {}
    with torch.no_grad():
        for conv in ("sft", "vllm"):
            cfg.monet.latent_convention = conv
            pb = PromptBuilder(proc, cfg.prompt, conv, K)
            inp = pb.build(views, instr).to(device)
            outs = {}
            for impl in ("cached", "recompute"):
                cfg.monet.recurrence_impl = impl
                r = MonetLatentReasoner(model, cfg.monet, cfg.conditioner)
                ts = time.time()
                outs[impl] = r(inp, st_tok)
                outs[impl].timings["wall_ms"] = (time.time() - ts) * 1000
            o = outs["cached"]
            c = {
                "input_len": int(inp.input_ids.shape[1]), "slot_start": inp.slot_start,
                "latents_shape": list(o.latents.shape), "vl_shape": list(o.vl_tokens.shape),
                "latents_finite": bool(torch.isfinite(o.latents).all()),
                "latent_norm_mean": float(o.latents.float().norm(dim=-1).mean()),
                "latent_norm_per_step": o.latents.float().norm(dim=-1).mean(0).tolist(),
                "self_feed_max_abs_err": float((o.fed_inputs[:, 1:] - o.latents[:, :-1]).abs().max()),
                "cached_vs_recompute_max_abs_err": float((o.latents - outs["recompute"].latents).abs().max()),
                "cos_consecutive_latents": [_cos(o.latents[:, j], o.latents[:, j + 1]) for j in range(K - 1)],
                "timings_ms": {k: v.timings for k, v in outs.items()},
            }
            r = MonetLatentReasoner(model, cfg.monet, cfg.conditioner)
            emb = r.embed(inp, st_tok)
            par = []
            for b in range(len(eps)):
                if conv == "sft":
                    fed, so, _ = authors_latent_mode_reference(model, inp, emb, b)
                    par.append({"vs": "authors Qwen2_5_VLModel.forward(latent_mode=True)",
                                "fed_max_abs_err": float((fed - o.fed_inputs[b]).abs().max()),
                                "slot_out_max_abs_err": float((so - o.latents[b]).abs().max()),
                                "rel_err": float((so - o.latents[b]).float().norm() / so.float().norm())})
                else:
                    fed, so = vllm_runner_emulation(r, inp, emb, b, K)
                    par.append({"vs": "vLLM GPUModelRunner ABS-VIS state machine (emulated)",
                                "fed_max_abs_err": float((fed - o.fed_inputs[b]).abs().max()),
                                "slot_out_max_abs_err": float((so - o.latents[b]).abs().max()),
                                "rel_err": float((so - o.latents[b]).float().norm() / so.float().norm())})
            c["parity"] = par
            # K = 0 vs K = 10: identical VL context, K=10 adds K new states
            cfg.monet.recurrence_impl = "cached"
            pb0 = PromptBuilder(proc, cfg.prompt, conv, 0)
            inp0 = pb0.build(views, instr).to(device)
            o0 = MonetLatentReasoner(model, cfg.monet, cfg.conditioner)(inp0, st_tok)
            P = min(o0.vl_tokens.shape[1], o.vl_tokens.shape[1])
            c["k0_latents_shape"] = list(o0.latents.shape)
            c["k0_vs_k10_vl_max_abs_err_on_shared_prefix"] = float((o0.vl_tokens[:, -P:-1] - o.vl_tokens[:, -P:-1]).abs().max()) if conv == "sft" else None
            c["cos_last_prompt_hidden_vs_latent_k"] = [_cos(o0.vl_tokens[:, -1], o.latents[:, j]) for j in range(K)]
            # not a soft prompt: feeding the same first vector K times (no recurrence) differs
            soft = MonetLatentReasoner(model, cfg.monet, cfg.conditioner)
            P2 = inp.slot_start
            seq = torch.cat([emb[:, :P2], o.fed_inputs[:, :1].expand(-1, K, -1)], 1)
            pos = soft.position_ids(inp)
            so_out = soft.lm(inputs_embeds=seq, position_ids=pos[:, :, :P2 + K], attention_mask=inp.attention_mask[:, :P2 + K]).last_hidden_state[:, P2:]
            c["self_fed_vs_repeated_soft_prompt_rel_diff"] = float((so_out - o.latents).float().norm() / o.latents.float().norm())
            # state dependence of the reasoning states
            st2 = proj((state + 1.0).to(device)).to(dtype)
            o2 = MonetLatentReasoner(model, cfg.monet, cfg.conditioner)(inp, st2)
            c["state_perturbation_latent_rel_change"] = float((o2.latents - o.latents).float().norm() / o.latents.float().norm())
            res[conv] = c
            lat_by_conv[conv] = o.latents
        # how much the authors' two latent conventions (HF SFT vs vLLM runner) disagree
        res["sft_vs_vllm_convention_cos_per_step"] = [
            _cos(lat_by_conv["sft"][:, j], lat_by_conv["vllm"][:, j]) for j in range(K)]
    rep["conventions"] = res
    return rep


def audit_data(cfg, data_root=None) -> Dict:
    from .libero_dataset import LeRobotLiberoFrames, LiberoWindowDataset, split_episodes
    from .normalization import ActionStateNormalizer

    frames = LeRobotLiberoFrames(data_root or cfg.data.root, file_limit=cfg.data.file_limit)
    tr, va = split_episodes(frames.episodes, cfg.data.val_fraction, cfg.data.split_seed)
    ds = LiberoWindowDataset(frames, tr, cfg.action_head.action_horizon)
    s = ds[len(ds) - 1]
    ep, st, ac = frames.episode_index, frames.state, frames.action
    same = np.r_[ep[1:] == ep[:-1], False]
    dq = np.r_[np.diff(st[:, 6]), 0]
    dpos = np.r_[st[1:, :3] - st[:-1, :3], np.zeros((1, 3))]
    a, sarr = ds.all_train_arrays()
    norm = ActionStateNormalizer.fit(a, sarr)
    na = norm.action.normalize(a)
    rt = norm.action.unnormalize(na)
    inside = (a[:, :6] >= norm.action.stats.q01[:6]) & (a[:, :6] <= norm.action.stats.q99[:6])
    return {
        "frames_loaded": int(len(ep)), "episodes": len(frames.episodes),
        "train_episodes": len(tr), "val_episodes": len(va),
        "train_val_disjoint": not (set(e.episode_index for e in tr) & set(e.episode_index for e in va)),
        "sample": {"images": [list(im.shape) for im in s["images"]], "image_dtype": str(s["images"][0].dtype),
                   "state": list(s["state"].shape), "actions": list(s["actions"].shape),
                   "loss_mask_last_window": s["loss_mask"].tolist(), "instruction": s["instruction"]},
        "gripper_values": np.unique(ac[:, 6]).tolist(),
        "gripper_close_cmd_dq_finger": float(dq[(ac[:, 6] == 1) & same].mean()),
        "gripper_open_cmd_dq_finger": float(dq[(ac[:, 6] == -1) & same].mean()),
        "translation_corr_action_vs_eef_delta": [float(np.corrcoef(ac[same, i], dpos[same, i])[0, 1]) for i in range(3)],
        "translation_mm_per_unit_action": [float(np.polyfit(ac[same, i], dpos[same, i], 1)[0] * 1000) for i in range(3)],
        "state_axis_angle_norm_range": [float(np.linalg.norm(st[:, 3:6], axis=1).min()), float(np.linalg.norm(st[:, 3:6], axis=1).max())],
        "action_q01": norm.action.stats.q01.tolist(), "action_q99": norm.action.stats.q99.tolist(),
        "normalisation_roundtrip_max_err_inside_q_range": float(np.abs(rt - a)[:, :6][inside].max()),
        "normalised_gripper_values": np.unique(na[:, 6]).tolist(),
    }


def audit_sim(cfg, data_root=None) -> Dict:

    from .libero_dataset import LeRobotLiberoFrames
    from .libero_env import LiberoEnv, libero_task_table

    frames = LeRobotLiberoFrames(data_root or cfg.data.root, file_limit=1)
    e0 = frames.episodes[0]
    table = libero_task_table(cfg.eval.suite)
    tid = [t for t, _, lang in table if lang == e0.instruction][0]
    env = LiberoEnv(cfg.eval.suite, tid, cfg.eval.camera_size, 0)
    t0 = time.time()
    obs = env.reset(0, cfg.eval.num_wait_steps)
    reset_ms = (time.time() - t0) * 1000
    ds_img = frames.image("image", e0.start).astype(np.float32)
    sim = obs.images["image"].astype(np.float32)
    corr = {"policy_view_rot180": float(np.corrcoef(sim.ravel(), ds_img.ravel())[0, 1]),
            "raw_unflipped": float(np.corrcoef(sim[::-1, ::-1].ravel(), ds_img.ravel())[0, 1])}
    s0 = obs.state.copy()
    t1 = time.time()
    a = np.array([1.0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)
    traj = []
    for _ in range(5):
        st = env.step(a)
        traj.append(st.observation.state.copy())
    step_ms = (time.time() - t1) * 1000 / 5
    dx = (traj[-1][:3] - s0[:3]) * 1000
    env.close()
    return {"task_id": tid, "instruction": e0.instruction, "reset_ms": reset_ms, "step_ms": step_ms,
            "image_corr_with_dataset": corr, "state_after_reset": s0.tolist(),
            "eef_displacement_mm_after_5x_plus_x": dx.tolist(), "num_init_states": int(len(env.init_states)),
            "max_steps": env.max_steps, "observation_refreshed": not np.allclose(traj[-1], s0)}


def render_markdown(rep: Dict) -> str:
    lines = ["# G0 audit report", "", f"Generated: {rep['generated']}", "", "## Pinned sources", ""]
    for k, (u, sha) in PINNED_SOURCES.items():
        lines.append(f"- **{k}**: {u} @ `{sha}`")
    lines += ["", "## Environment", "", "```json", json.dumps(rep["environment"], indent=2), "```", ""]
    for sec in ("monet", "data", "sim"):
        if sec in rep:
            lines += [f"## {sec}", "", "```json", json.dumps(rep[sec], indent=2, default=str), "```", ""]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--out", default="results/audit")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tokenizer-path", default=None)
    ap.add_argument("--skip-monet", action="store_true")
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.override)
    rep: Dict = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "environment": environment_report(),
                 "pinned_sources": PINNED_SOURCES}
    if not a.skip_data:
        rep["data"] = audit_data(cfg)
    if not a.skip_monet:
        rep["monet"] = audit_monet(cfg, a.tokenizer_path, a.device)
    if not a.skip_sim:
        rep["sim"] = audit_sim(cfg)
    out = Path(a.out)
    tag = "tiny" if cfg.monet.model_path == "tiny-random" else "monet7b"
    write_json(out / f"g0_report_{tag}.json", rep)
    (out / f"g0_report_{tag}.md").write_text(render_markdown(rep))
    print(json.dumps({k: v for k, v in rep.items() if k != "environment"}, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
