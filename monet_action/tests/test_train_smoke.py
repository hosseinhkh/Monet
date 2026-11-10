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
