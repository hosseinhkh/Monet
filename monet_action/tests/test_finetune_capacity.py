"""Monet adaptation capacity ladder (docs/FINETUNING.md): wide LoRA (attention + MLP) and full fine-tuning of
whole decoder layers, on the tiny Monet (the authors' real modeling code, 2 layers)."""
import pytest
import torch

from monet_action.config import load_config
from monet_action.monet_loader import build_tiny_monet
from monet_action.policy import MonetActionPolicy

from conftest import INSTR, ROOT, random_views

pytestmark = pytest.mark.tokenizer


def _batch(policy, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    inputs = policy.prompt_builder.build(random_views(B, seed=seed), INSTR[:B])
    return {"monet_inputs": inputs, "state": torch.randn(B, 8, generator=g),
            "actions": torch.rand(B, 8, 7, generator=g) * 2 - 1, "loss_mask": torch.ones(B, 8)}


def _policy(cfg_path, processor, overrides=(), dtype=None):
    cfg = load_config(cfg_path, list(overrides))
    model = build_tiny_monet(seed=0)
    if dtype is not None:
        model = model.to(dtype)
    return MonetActionPolicy(cfg, model, processor), cfg


def test_full_finetune_layer_trains_through_the_latent_loop(cfg_path, processor):
    from monet_action.train import GRAD_GROUPS, assert_gradients, grad_report, param_groups

    pol, cfg = _policy(cfg_path, processor, ["monet.finetune_layers=[1]", "train.lr_backbone=1e-5"])
    layer1 = "reasoner.model.model.language_model.layers.1."
    trainable = {n for n, p in pol.named_parameters() if p.requires_grad and n.startswith("reasoner.model.")}
    assert trainable and all(n.startswith(layer1) for n in trainable)
    assert all(p.dtype == torch.float32 for n, p in pol.named_parameters() if n in trainable)
    rep = pol.parameter_report()
    layer_params = sum(p.numel() for p in pol.reasoner.model.model.language_model.layers[1].parameters())
    assert rep["monet_full"] == layer_params and rep["monet_lora"] == 0
    groups = {g["name"]: g for g in param_groups(pol, cfg.train)}
    assert groups["backbone"]["lr"] == 1e-5 and groups["backbone"]["weight_decay"] == 0.0
    assert sum(p.numel() for p in groups["backbone"]["params"]) == layer_params

    pol.train()
    loss, _ = pol.compute_loss(_batch(pol), generator=torch.Generator().manual_seed(0))
    loss.backward()
    g = grad_report(pol)
    assert g["monet_full"]["grad_norm"] > 0 and g["monet_full"]["finite"]
    assert_gradients(g, cfg)  # monet_full is required when finetune_layers is set
    for n, p in pol.named_parameters():
        if n.startswith("reasoner.model.") and n not in trainable:
            assert p.grad is None, f"frozen Monet weight received a gradient: {n}"
    assert "monet_full" in GRAD_GROUPS


def test_full_finetune_reaches_layer0_only_through_the_recurrence(cfg_path, processor):
    """Layer 0 sits before the last layer; with K>0 it still gets gradient (latents are re-fed through it)."""
    pol, _ = _policy(cfg_path, processor, ["monet.finetune_layers=[0]"])
    pol.train()
    loss, _ = pol.compute_loss(_batch(pol), generator=torch.Generator().manual_seed(0))
    loss.backward()
    grads = [p.grad for n, p in pol.named_parameters() if n.startswith("reasoner.model.") and p.requires_grad]
    assert grads and all(gr is not None and torch.isfinite(gr).all() for gr in grads)
    assert sum(float(gr.abs().sum()) for gr in grads) > 0


def test_finetuned_layers_are_saved_and_restored(cfg_path, processor):
    pol, _ = _policy(cfg_path, processor, ["monet.finetune_layers=[1]"])
    opt = torch.optim.SGD([p for p in pol.parameters() if p.requires_grad], lr=1e-1)
    pol.train()
    loss, _ = pol.compute_loss(_batch(pol), generator=torch.Generator().manual_seed(0))
    loss.backward()
    opt.step()
    sd = pol.trainable_state_dict()
    assert any(k.startswith("reasoner.model.model.language_model.layers.1.") for k in sd)
    fresh, _ = _policy(cfg_path, processor, ["monet.finetune_layers=[1]"])
    fresh.load_trainable_state_dict(sd, strict=True)
    a = dict(pol.named_parameters())
    for n, p in fresh.named_parameters():
        assert torch.equal(p, a[n]), n


def test_wide_lora_covers_attention_and_mlp(cfg_path, processor):
    pol, _ = _policy(cfg_path, processor, ["monet.lora_enabled=true", "monet.lora_layers=[0,1]", "monet.lora_r=4",
                                           "monet.lora_alpha=8",
                                           "monet.lora_target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"])
    names = [n for n, _ in pol.named_parameters() if "lora_A" in n]
    for mod in ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        assert sum(mod in n for n in names) == 2, mod  # both layers
    assert len(names) == 14


def test_capacity_configs_validate():
    from monet_action.config import load_config as lc

    for k10, k0 in (("d2_lora_wide", "d0_k0_lora_wide"), ("d2_ft4", "d0_k0_ft4"), ("d2_ft8", "d0_k0_ft8")):
        a, b = lc(str(ROOT / "configs" / f"{k10}.yaml")), lc(str(ROOT / "configs" / f"{k0}.yaml"))
        assert (a.monet.num_latents, b.monet.num_latents) == (10, 0)
        for f in ("lora_enabled", "lora_r", "lora_alpha", "lora_layers", "lora_target_modules", "finetune_layers"):
            assert getattr(a.monet, f) == getattr(b.monet, f), (k10, f)  # matched adaptation
        assert a.train.max_steps == b.train.max_steps == 10000
        assert a.train.init_from.endswith("d1_k10/ckpt_0010000") and b.train.init_from.endswith("d0_k0/ckpt_0010000")
    with pytest.raises(ValueError, match="both LoRA-adapted and fully fine-tuned"):
        lc(str(ROOT / "configs" / "d2_lora.yaml"), ["monet.finetune_layers=[27]"])
    with pytest.raises(ValueError, match="distinct"):
        lc(str(ROOT / "configs" / "d1_k10.yaml"), ["monet.finetune_layers=[3,3]"])


def test_out_of_range_finetune_layer_is_rejected(cfg_path, processor):
    with pytest.raises(ValueError, match="out of range"):
        _policy(cfg_path, processor, ["monet.finetune_layers=[5]"])  # the tiny model has 2 layers


def test_bf16_model_with_fp32_finetuned_layer_under_autocast(cfg_path, processor):
    """The real setting: bf16 Monet, one fp32 trainable layer, bf16 autocast. No dtype clash; latents stay bf16;
    the trainable layer gets fp32 gradients (CPU autocast stands in for CUDA autocast here)."""
    pol, _ = _policy(cfg_path, processor, ["monet.finetune_layers=[1]"], dtype=torch.bfloat16)
    pol.train()
    b = _batch(pol)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cond, aux = pol.encode(b["monet_inputs"], b["state"])
    assert aux.latents.dtype == torch.bfloat16
    aux.latents.float().pow(2).mean().backward()  # the latents pass through the trainable layer K times
    ps = [p for n, p in pol.named_parameters() if n.startswith("reasoner.model.") and p.requires_grad]
    assert ps and all(p.dtype == torch.float32 and p.grad is not None and p.grad.dtype == torch.float32 for p in ps)
    assert sum(float(p.grad.abs().sum()) for p in ps) > 0


def test_numeric_overrides_are_numbers():
    """YAML reads 1e-5 as a string; the loader converts it by the field type (once broke lr overrides)."""
    c = load_config(str(ROOT / "configs" / "d2_ft4.yaml"), ["train.lr_backbone=1e-5", "train.lr_head=2e-4",
                                                           "eval.gpu_hourly_cost_usd=2", "train.max_steps=100"])
    assert (c.train.lr_backbone, c.train.lr_head, c.eval.gpu_hourly_cost_usd, c.train.max_steps) == (1e-5, 2e-4, 2.0, 100)
    assert isinstance(c.train.lr_backbone, float) and isinstance(c.eval.gpu_hourly_cost_usd, float)
    with pytest.raises(ValueError, match="expects a number"):
        load_config(str(ROOT / "configs" / "d2_ft4.yaml"), ["train.lr_head=fast"])


def _fake_eval(d, successes):
    """Minimal evaluate.py output: episodes.jsonl + summary.json from metrics.aggregate_episodes."""
    import json

    from monet_action.metrics import aggregate_episodes

    recs = [{"task_id": i % 2, "init_state_idx": i, "seed": 0, "success": bool(s), "env_steps": 10, "action_calls": 10,
             "sim_time_ms": 100.0, "batch_sizes": [1], "call_latency_ms": [{"monet_prefill": 5.0, "action_head": 2.0}],
             "failure_tags": [], "wall_s": 1.0} for i, s in enumerate(successes)]
    d.mkdir(parents=True)
    (d / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    (d / "summary.json").write_text(json.dumps({**aggregate_episodes(recs), "task_names": {}}))


def test_report_capacity_section(tmp_path):
    from monet_action.report import build_report

    _fake_eval(tmp_path / "d1_k10", [1, 0, 1, 0])
    _fake_eval(tmp_path / "d0_k0", [0, 0, 1, 0])
    _fake_eval(tmp_path / "d2_ft4", [1, 1, 1, 0])
    _fake_eval(tmp_path / "d0_k0_ft4", [1, 0, 0, 0])
    md = build_report(str(tmp_path))
    sec = md[md.index("## 4d."):md.index("## 4c.")]
    assert "| frozen | none | 0.500 | 0.250 | +0.250 |" in sec
    assert "| full FT, 4 layers | layers 24-27 in full (~0.93B) | 0.750 | 0.250 | +0.500 |" in sec
    assert "| wide LoRA |" in sec and "not run" in sec
    assert "Paired statistics, full FT, 4 layers:" in sec and "McNemar" in sec


def test_recommendation_file_per_profile(tmp_path):
    import json

    from monet_action import preflight

    assert preflight.recommended_name(tmp_path / "memory_profile.json") == "recommended_batch.txt"
    assert preflight.recommended_name(tmp_path / "memory_profile_d2_ft4.json") == "recommended_batch_d2_ft4.txt"
    p = tmp_path / "memory_profile_d2_ft4.json"
    p.write_text(json.dumps([{"batch_size": 2, "ok": True, "peak_gb": 40.0, "step_time_s": 2.0, "oom": False, "error": None},
                             {"batch_size": 4, "ok": False, "peak_gb": None, "step_time_s": None, "oom": True, "error": "OOM"}]))
    assert preflight.check_memory(str(p), budget_gb=44.5) == 0
    assert (tmp_path / "recommended_batch_d2_ft4.txt").read_text().strip() == "train.batch_size=2 train.grad_accum=8"
    assert not (tmp_path / "recommended_batch.txt").exists()  # the D2 recommendation is left alone


def test_train_capacity_script_rejects_unknown_level():
    import subprocess

    from conftest import common_sh_env

    r = subprocess.run(["bash", "scripts/train_capacity.sh", "ft99"], cwd=ROOT, env={**common_sh_env(), "MONET_GPU_HOLD": "0"},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "unknown level" in r.stdout
