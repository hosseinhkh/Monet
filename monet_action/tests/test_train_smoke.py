"""G1 overfit + checkpoint reload, on the tiny Monet (real Monet code path) and real LIBERO frames."""
import json

import numpy as np
import pytest
import torch

from monet_action.config import load_config
from monet_action.train import load_policy_from_checkpoint, train

pytestmark = [pytest.mark.slow, pytest.mark.tokenizer]


def test_tiny_overfit_and_reload(cfg_path, data_root, processor, tmp_path):
    out = tmp_path / "run"
    cfg = load_config(cfg_path, [f"data.root={data_root}", f"train.output_dir={out}", "train.max_steps=300",
                                 "train.log_every=10", "data.max_windows=32", "train.batch_size=8",
                                 "train.overfit_eval_every=100"])
    train(cfg, device="cpu")
    log = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines() if '"loss"' in line]
    losses = [r["loss"] for r in log]
    assert np.mean(losses[-3:]) < 0.35 * losses[0], losses
    # deterministic (fixed t / noise) metrics on the overfit windows decrease monotonically
    ov = [json.loads(line) for line in (out / "overfit_log.jsonl").read_text().splitlines()]
    fm = [r["fm_loss"] for r in ov]
    assert all(b < a for a, b in zip(fm, fm[1:])), fm
    audit = json.loads((out / "gradient_audit.json").read_text())
    assert audit["passed"] and audit["report"]["state_projector"]["grad_norm"] > 0

    policy, cfg2, meta = load_policy_from_checkpoint(str(out), "cpu")
    assert meta["step"] == 300
    from monet_action.libero_dataset import Collator, build_datasets

    tr, _, norm, _ = build_datasets(cfg2.data, 8, cfg2.tool_result.num_prev_actions)
    idx = list(range(0, 32, 4))
    batch = Collator(policy.prompt_builder, norm)([tr[i] for i in idx])
    noise = torch.randn(len(idx), 8, 7, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        cond, _ = policy.encode(batch["monet_inputs"], batch["state"], prev_actions=batch["prev_actions"],
                                prev_valid=batch["prev_valid"])
        pred = policy.action_head.sample(cond.tokens, cond.mask, batch["state"], num_steps=4, noise=noise)
    tgt = batch["actions"]
    mse = float(((pred - tgt) ** 2).mean())
    assert mse < 0.25 * float((tgt ** 2).mean())  # far better than a zero/no-op prediction
    # non-degenerate: predictions vary across inputs about as much as the targets do
    assert float(pred[..., :6].std(0).mean()) > 0.25 * float(tgt[..., :6].std(0).mean())
    # reloaded checkpoint reproduces the in-memory model's final open-loop metric exactly
    ov_last = ov[-1]
    from torch.utils.data import DataLoader

    from monet_action.metrics import offline_metrics

    dl = DataLoader(tr, batch_size=8, shuffle=False, collate_fn=Collator(policy.prompt_builder, norm))
    om = offline_metrics(policy, dl, max_batches=1000, num_steps=4)
    assert abs(om["fm_loss"] - ov_last["fm_loss"]) < 1e-4


def test_gpu_summary_reaches_status_and_events(cfg_path, data_root, processor, tmp_path, monkeypatch):
    """CPU runs never take train()'s CUDA branch; feed it what setup_gpu returns on the server (the summary has
    a ``name`` key - it once collided with RunMonitor.event(name, ...) and crashed every GPU run at start-up)."""
    import monet_action.train as T

    fake = {"cuda": True, "name": "NVIDIA RTX PRO 6000 Blackwell", "capability": "sm_120", "total_gb": 95.0,
            "free_gb": 20.0, "visible_devices": 1, "CUDA_VISIBLE_DEVICES": "0", "MUJOCO_EGL_DEVICE_ID": "0",
            "reserved_by_this_process_gb": 46.5, "reserve_fraction": 0.5, "reserved_now_gb": 46.5,
            "limit_fraction": 0.5, "limit_gb": 47.5, "holder_lease": True}
    monkeypatch.setattr(T, "setup_gpu", lambda device: dict(fake))
    out = tmp_path / "run"
    cfg = load_config(cfg_path, [f"data.root={data_root}", f"train.output_dir={out}", "train.max_steps=2",
                                 "data.max_windows=8", "train.batch_size=4", "train.overfit_eval_every=0"])
    train(cfg, device="cpu")
    ev = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    g = [e for e in ev if e["event"] == "gpu"]
    assert g and g[0]["name"] == fake["name"] and g[0]["holder_lease"] is True
    assert json.loads((out / "status.json").read_text())["gpu"]["name"] == fake["name"]


def test_full_finetune_stage_b_from_frozen_stage_a(cfg_path, data_root, processor, tmp_path):
    """The D2-FT4 / D2-FT8 path end to end: frozen stage A -> stage B that fully fine-tunes a decoder layer,
    warm-started from stage A (as d2_ft4 starts from D1@10k); gradient audit, checkpoint, reload for eval."""
    common = [f"data.root={data_root}", "data.max_windows=8", "train.batch_size=4", "train.overfit_eval_every=0",
              "train.max_steps=2", "train.save_every=2", "train.warmup_steps=1"]
    a, b = tmp_path / "stage_a", tmp_path / "stage_b"
    train(load_config(cfg_path, common + [f"train.output_dir={a}"]), device="cpu")
    cfg_b = load_config(cfg_path, common + [f"train.output_dir={b}", f"train.init_from={a}", "monet.finetune_layers=[1]",
                                            "train.lr_backbone=1e-3"])
    train(cfg_b, device="cpu")
    audit = json.loads((b / "gradient_audit.json").read_text())
    assert audit["passed"] and audit["report"]["monet_full"]["grad_norm"] > 0
    rep = json.loads((b / "param_report.json").read_text())
    assert rep["monet_full"] > 0 and rep["monet_lora"] == 0
    ev = [json.loads(line)["event"] for line in (b / "events.jsonl").read_text().splitlines()]
    assert "init_from" in ev
    log = [json.loads(line) for line in (b / "train_log.jsonl").read_text().splitlines() if '"lrs"' in line]
    assert "backbone" in log[-1]["lrs"]
    policy, cfg2, meta = load_policy_from_checkpoint(str(b), "cpu")
    assert cfg2.monet.finetune_layers == [1] and meta["step"] == 2
    sd = torch.load(b / (b / "last").read_text().strip() / "trainable.pt", map_location="cpu")
    w = "reasoner.model.model.language_model.layers.1.mlp.down_proj.weight"
    assert w in sd and torch.equal(dict(policy.named_parameters())[w].detach(), sd[w])
    fresh = build_tiny_monet_layer1()
    assert not torch.equal(fresh, sd[w])  # stage B actually moved the pretrained layer


def build_tiny_monet_layer1():
    from monet_action.monet_loader import build_tiny_monet

    return build_tiny_monet(seed=0).model.language_model.layers[1].mlp.down_proj.weight.detach()
