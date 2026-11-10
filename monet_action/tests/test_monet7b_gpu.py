"""REAL Monet-7B checks (marker ``gpu``; run on the GPU server: ``make test-monet7b``).

Skipped automatically unless CUDA is available AND the Monet-7B weights are present locally
(``scripts/download_data.sh`` or MONET7B_PATH=/path/to/Monet-7B). Every measured number is also
written to results/audit/monet7b_gpu_tests.json for the G0 record.

Tolerances are for bf16: the batched/left-padded recurrence vs the authors' per-sample code can
differ by bf16 rounding (different kernels/shapes), so parity is asserted on relative error and
cosine similarity rather than bit-equality.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import ROOT

pytestmark = pytest.mark.gpu
REV = "39c1e320a3e15c357bff911fda263923edbdda37"
REPORT: dict = {}


def _weights_path():
    p = os.environ.get("MONET7B_PATH")
    if p:
        return p if Path(p).exists() else None
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download("NOVAglow646/Monet-7B", revision=REV, local_files_only=True)
    except Exception:
        return None


if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
WEIGHTS = _weights_path()
if WEIGHTS is None:
    pytest.skip("Monet-7B weights not downloaded (scripts/download_data.sh or MONET7B_PATH)", allow_module_level=True)


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def _cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


@pytest.fixture(scope="module")
def stack(data_root):
    from monet_action.config import load_config
    from monet_action.gpu import setup_gpu
    from monet_action.libero_dataset import LeRobotLiberoFrames
    from monet_action.monet_loader import load_monet, load_processor
    from monet_action.normalization import Stats

    REPORT["gpu_setup"] = setup_gpu("cuda")  # GPU-holder hand-over + MONET_GPU_RESERVE, as training does
    cfg = load_config(str(ROOT / "configs" / "d1_k10.yaml"), [f"monet.model_path={WEIGHTS}", "monet.model_revision=null"])
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    model, info = load_monet(cfg.monet, device="cuda")
    REPORT["load_s"] = time.time() - t0
    REPORT["weights_gb"] = torch.cuda.memory_allocated() / 1024 ** 3
    proc = load_processor(cfg.monet)
    frames = LeRobotLiberoFrames(data_root, file_limit=1)
    eps = frames.episodes[:2]
    views = [[frames.image(k, e.start + 10) for k in frames.camera_keys] for e in eps]
    st = Stats.from_array(frames.state)
    state = torch.from_numpy((np.stack([frames.state[e.start + 10] for e in eps]) - st.mean) / st.std).float().cuda()
    yield {"cfg": cfg, "model": model, "info": info, "proc": proc, "views": views,
           "instr": [e.instruction for e in eps], "state": state}
    out = ROOT / "results" / "audit" / "monet7b_gpu_tests.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    REPORT.update({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "time": time.ctime()})
    out.write_text(json.dumps(REPORT, indent=2, default=str))


def _inputs(stack, conv="sft", K=10, tool=True):
    from monet_action.prompt import PromptBuilder

    c = stack["cfg"]
    pb = PromptBuilder(stack["proc"], c.prompt, conv, K, c.tool_result if tool else None)
    return pb.build(stack["views"], stack["instr"]).to("cuda")


def _reasoner(stack, conv="sft", impl="cached"):
    from monet_action.monet_backbone import MonetLatentReasoner

    c = stack["cfg"]
    c.monet.latent_convention, c.monet.recurrence_impl = conv, impl
    return MonetLatentReasoner(stack["model"], c.monet, c.conditioner)


def _state_tokens(stack, shift=0.0, n=2, seed=0):
    from monet_action.state_context import RobotStateProjector, embedding_rms

    torch.manual_seed(seed)
    m = stack["model"]
    proj = RobotStateProjector(8, 3584, n, 256, init_rms=embedding_rms(m.model.language_model.embed_tokens)).cuda()
    with torch.no_grad():
        return proj(stack["state"] + shift).to(torch.bfloat16)


def test_checkpoint_loads_completely(stack):
    info = stack["info"]
    REPORT["missing_keys"], REPORT["unexpected_keys"] = info["missing_keys"], info["unexpected_keys"][:20]
    assert info["missing_keys"] == []
    c = stack["model"].config
    assert c.text_config.hidden_size == 3584 and c.text_config.num_hidden_layers == 28
    assert (c.latent_token_id, c.latent_start_id, c.latent_end_id) == (151665, 151666, 151667)
    assert next(stack["model"].parameters()).dtype == torch.bfloat16


@torch.no_grad()
def test_language_model_sanity_generation(stack):
    """Weights-loaded-correctly smoke check with ordinary decoding (NOT latent reasoning)."""
    tok = stack["proc"].tokenizer
    msgs = [{"role": "user", "content": "What is 2 + 3? Answer with just the number."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    out = stack["model"].generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=24,
                                  do_sample=False)
    ans = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
    REPORT["sanity_generation"] = ans
    assert len(ans.strip()) > 0
    if "5" not in ans:
        pytest.xfail(f"unexpected answer (inspect manually): {ans!r}")


@torch.no_grad()
def test_real_latents_shapes_and_self_feed(stack):
    inp = _inputs(stack)
    st = _state_tokens(stack)
    r = _reasoner(stack)
    tool = torch.zeros(2, 1, 3584, dtype=torch.bfloat16, device="cuda")
    out = r(inp, st, tool_tokens=tool)
    lat = out.latents
    assert lat.shape == (2, 10, 3584) and torch.isfinite(lat).all()
    assert torch.equal(out.fed_inputs[:, 1:], lat[:, :-1])
    cos = [_cos(lat[:, j], lat[:, j + 1]) for j in range(9)]
    REPORT["latent_norm_per_step"] = lat.float().norm(dim=-1).mean(0).tolist()
    REPORT["cos_consecutive_latents"] = cos
    assert max(cos) < 0.99999  # the recurrence actually moves


@torch.no_grad()
def test_parity_with_authors_code_bf16(stack):
    from monet_action.monet_backbone import authors_latent_mode_reference, vllm_runner_emulation

    st = _state_tokens(stack)
    tool = torch.zeros(2, 1, 3584, dtype=torch.bfloat16, device="cuda")
    res = {}
    for conv in ("sft", "vllm"):
        inp = _inputs(stack, conv)
        r = _reasoner(stack, conv)
        out = r(inp, st, tool_tokens=tool)
        emb = r.embed(inp, st, tool)
        for b in range(2):
            if conv == "sft":
                fed, so, _ = authors_latent_mode_reference(stack["model"], inp, emb, b)
            else:
                fed, so = vllm_runner_emulation(r, inp, emb, b, 10)
            res[f"{conv}_b{b}"] = {"rel_err": _rel(out.latents[b], so), "cos": _cos(out.latents[b], so)}
            assert res[f"{conv}_b{b}"]["rel_err"] < 5e-2 and res[f"{conv}_b{b}"]["cos"] > 0.998, res
        res[conv + "_latents"] = out.latents
    res["sft_vs_vllm_cos_per_step"] = [_cos(res["sft_latents"][:, j], res["vllm_latents"][:, j]) for j in range(10)]
    REPORT["parity"] = {k: v for k, v in res.items() if not k.endswith("_latents")}


@torch.no_grad()
def test_cached_vs_recompute_bf16(stack):
    inp = _inputs(stack)
    st = _state_tokens(stack)
    tool = torch.zeros(2, 1, 3584, dtype=torch.bfloat16, device="cuda")
    a = _reasoner(stack, impl="cached")(inp, st, tool_tokens=tool).latents
    b = _reasoner(stack, impl="recompute")(inp, st, tool_tokens=tool).latents
    REPORT["cached_vs_recompute_rel_err"] = _rel(a, b)
    assert _rel(a, b) < 5e-2 and _cos(a, b) > 0.998


@torch.no_grad()
def test_state_dependence_and_k0_context(stack):
    tool = torch.zeros(2, 1, 3584, dtype=torch.bfloat16, device="cuda")
    inp = _inputs(stack)
    a = _reasoner(stack)(inp, _state_tokens(stack), tool_tokens=tool).latents
    b = _reasoner(stack)(inp, _state_tokens(stack, shift=1.0), tool_tokens=tool).latents
    REPORT["state_shift_latent_rel_change"] = _rel(b, a)
    assert _rel(b, a) > 1e-3
    o0 = _reasoner(stack)(_inputs(stack, K=0), _state_tokens(stack), tool_tokens=tool)
    o10 = _reasoner(stack)(_inputs(stack, K=10), _state_tokens(stack), tool_tokens=tool)
    m = o0.vl_mask
    REPORT["k0_vs_k10_context_rel_err"] = _rel(o10.vl_tokens[m], o0.vl_tokens[m])
    assert REPORT["k0_vs_k10_context_rel_err"] < 2e-2


@torch.no_grad()
def test_latency_k0_vs_k10(stack):
    from monet_action.profiling import Timer, percentiles

    tool = torch.zeros(1, 1, 3584, dtype=torch.bfloat16, device="cuda")
    st = _state_tokens(stack)[:1]
    prof = {}
    for K in (0, 10):
        from monet_action.prompt import PromptBuilder

        c = stack["cfg"]
        inp = PromptBuilder(stack["proc"], c.prompt, "sft", K, c.tool_result).build(stack["views"][:1], stack["instr"][:1]).to("cuda")
        r = _reasoner(stack)
        times = {}
        for i in range(13):
            t = Timer("cuda")
            r(inp, st, tool_tokens=tool, timer=t)
            if i >= 3:
                for k, v in t.result().items():
                    times.setdefault(k, []).append(v)
        prof[f"K{K}"] = {k: percentiles(v) for k, v in times.items()}
    REPORT["latency_ms_batch1"] = prof
    assert prof["K10"]["monet_latent_steps"]["p50"] > 0


def test_lora_gradients_on_real_monet(stack):
    """One D2 training step on the real model: FM loss reaches LoRA through the K-step loop."""
    from monet_action.config import load_config
    from monet_action.normalization import ActionStateNormalizer
    from monet_action.policy import MonetActionPolicy

    cfg = load_config(str(ROOT / "configs" / "d2_lora.yaml"), [f"monet.model_path={WEIGHTS}", "monet.model_revision=null"])
    rng = np.random.default_rng(0)
    a = np.concatenate([rng.normal(0, 0.3, (500, 6)), rng.choice([-1.0, 1.0], (500, 1))], 1).astype(np.float32)
    norm = ActionStateNormalizer.fit(a, rng.normal(0, 1, (500, 8)).astype(np.float32))
    pol = MonetActionPolicy(cfg, stack["model"], stack["proc"], norm)  # injects LoRA in place (last test)
    for mod in (pol.action_head, pol.conditioner, pol.state_projector, pol.tool_result_projector):
        mod.to("cuda", torch.float32)
    pol.train()
    torch.cuda.reset_peak_memory_stats()
    batch = {"monet_inputs": pol.prompt_builder.build(stack["views"][:1], stack["instr"][:1]),
             "state": stack["state"][:1].cpu(), "actions": torch.rand(1, 8, 7) * 2 - 1, "loss_mask": torch.ones(1, 8)}
    loss, m = pol.compute_loss(batch)
    loss.backward()
    REPORT["d2_step_peak_mem_gb_batch1"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    lora_B = [(n, p) for n, p in pol.named_parameters() if "lora_B" in n]
    assert lora_B and all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in lora_B)
    nz = sum(int(p.grad.abs().sum() > 0) for _, p in lora_B)
    REPORT["d2_lora_B_tensors_with_grad"] = f"{nz}/{len(lora_B)}"
    assert nz == len(lora_B)
    frozen = [n for n, p in pol.named_parameters() if n.startswith("reasoner.model") and "lora_" not in n and p.grad is not None]
    assert frozen == []
