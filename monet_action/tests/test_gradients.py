"""Critical test 4: FM loss -> DiT -> conditioner -> latent recurrence -> selected Monet LoRA."""
import pytest
import torch

from monet_action.config import load_config
from monet_action.monet_backbone import MonetLatentReasoner
from monet_action.monet_loader import build_tiny_monet
from monet_action.policy import MonetActionPolicy

from conftest import INSTR, build_inputs, random_views

pytestmark = pytest.mark.tokenizer


def _batch(policy, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    inputs = policy.prompt_builder.build(random_views(B, seed=seed), INSTR[:B])
    return {"monet_inputs": inputs, "state": torch.randn(B, 8, generator=g),
            "actions": torch.rand(B, 8, 7, generator=g) * 2 - 1, "loss_mask": torch.ones(B, 8)}


def _policy(cfg_path, processor, overrides=()):
    cfg = load_config(cfg_path, list(overrides))
    model = build_tiny_monet(seed=0)
    return MonetActionPolicy(cfg, model, processor), cfg


def test_fm_gradient_reaches_lora_through_latent_loop(cfg_path, processor):
    pol, cfg = _policy(cfg_path, processor, ["monet.lora_enabled=true", "monet.lora_layers=[0,1]",
                                             "monet.lora_target_modules=[q_proj,v_proj]"])
    pol.train()
    loss, _ = pol.compute_loss(_batch(pol), generator=torch.Generator().manual_seed(0))
    loss.backward()
    groups = {"action_head": [], "conditioner": [], "state_projector": [], "lora": [], "frozen": []}
    for n, p in pol.named_parameters():
        key = ("lora" if "lora_" in n else "frozen") if n.startswith("reasoner.model") else n.split(".")[0]
        if key in groups:
            groups[key].append((n, p))
    for k in ("action_head", "conditioner", "state_projector", "lora"):
        assert groups[k], k
        for n, p in groups[k]:
            if k == "lora" and "lora_A" in n:
                continue  # lora_B is zero-initialised, so lora_A grads are exactly 0 at step 0
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
        assert sum(float(p.grad.abs().sum()) for n, p in groups[k] if p.grad is not None) > 0, k
    for n, p in groups["frozen"]:
        assert p.grad is None, f"frozen Monet weight received a gradient: {n}"
    lora_B = [p for n, p in groups["lora"] if "lora_B" in n]
    assert len(lora_B) == 4 and all(float(p.grad.abs().sum()) > 0 for p in lora_B)


def test_lora_A_gets_gradient_after_one_step(cfg_path, processor):
    pol, _ = _policy(cfg_path, processor, ["monet.lora_enabled=true", "monet.lora_layers=[1]"])
    pol.train()
    opt = torch.optim.SGD([p for p in pol.parameters() if p.requires_grad], lr=1e-2)
    for i in range(2):
        opt.zero_grad()
        loss, _ = pol.compute_loss(_batch(pol, seed=i), generator=torch.Generator().manual_seed(i))
        loss.backward()
        opt.step()
    lora_A = [(n, p) for n, p in pol.named_parameters() if "lora_A" in n]
    assert all(float(p.grad.abs().sum()) > 0 and torch.isfinite(p.grad).all() for _, p in lora_A)


def test_bptt_vs_truncated_feedback(tiny_model, processor, cfg):
    """With bptt the last latent depends on the first fed input through the recurrence; the
    authors' truncated (detached) feedback cuts exactly that path."""
    inp, _ = build_inputs(processor, cfg, "sft")
    grads = {}
    for mode, expect_grad in (("bptt", True), ("truncated", False)):
        cfg.monet.feedback_grad = mode
        r = MonetLatentReasoner(tiny_model, cfg.monet, cfg.conditioner)
        st = torch.randn(2, 2, 64, generator=torch.Generator().manual_seed(0)).requires_grad_(True)
        out = r(inp, st)
        assert out.fed_inputs[:, 1:].requires_grad == expect_grad
        if expect_grad:
            assert out.fed_inputs[:, 1].grad_fn is not None  # no hidden .detach() at the boundary
        (g,) = torch.autograd.grad(out.latents[:, -1].sum(), st)
        assert torch.isfinite(g).all() and g.abs().sum() > 0  # KV-cache path exists in both modes
        grads[mode] = g
    # the recurrent (fed-back) path contributes extra gradient only under bptt
    assert (grads["bptt"] - grads["truncated"]).abs().max() > 1e-6


def test_frozen_variant_trains_only_head_and_projector(cfg_path, processor):
    pol, _ = _policy(cfg_path, processor)
    trainable = {n.split(".")[0] for n, p in pol.named_parameters() if p.requires_grad}
    assert trainable == {"state_projector", "tool_result_projector", "conditioner", "action_head"}


def test_d0_and_d1_have_identical_trainable_capacity(cfg_path, processor):
    p0, _ = _policy(cfg_path, processor, ["monet.num_latents=0"])
    p1, _ = _policy(cfg_path, processor, ["monet.num_latents=10"])
    c0 = sum(p.numel() for p in p0.parameters() if p.requires_grad)
    c1 = sum(p.numel() for p in p1.parameters() if p.requires_grad)
    assert c0 == c1
