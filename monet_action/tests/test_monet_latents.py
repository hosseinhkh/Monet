"""Critical tests 1-3: genuine Monet self-fed recurrence, train/inference parity, state dependence.

All tests run Monet's *actual* customised Qwen2.5-VL code (monet_qwen_model/...) with a tiny
random configuration; they validate the mechanism, not Monet-7B's weights (that is G0 on GPU).
"""
import pytest
import torch

from monet_action.monet_backbone import MonetLatentReasoner, authors_latent_mode_reference, vllm_runner_emulation

from conftest import build_inputs

pytestmark = pytest.mark.tokenizer
ATOL = 2e-5


def _reasoner(model, cfg, impl="cached", grad="bptt", mode="self_fed"):
    cfg.monet.recurrence_impl, cfg.monet.feedback_grad, cfg.monet.reasoning_mode = impl, grad, mode
    return MonetLatentReasoner(model, cfg.monet, cfg.conditioner)


def _state_tokens(model, B=2, n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, n, model.config.text_config.hidden_size, generator=g)


@pytest.mark.parametrize("conv", ["sft", "vllm"])
def test_self_feed_property(tiny_model, processor, cfg, conv):
    inp, _ = build_inputs(processor, cfg, conv)
    out = _reasoner(tiny_model, cfg)(inp, _state_tokens(tiny_model))
    assert out.latents.shape == (2, 10, 64) and out.fed_inputs.shape == (2, 10, 64)
    assert torch.isfinite(out.latents).all()
    # input of slot k+1 is *exactly* the final hidden state produced at slot k
    assert torch.equal(out.fed_inputs[:, 1:], out.latents[:, :-1])
    # slot 0 is fed the last prompt hidden state
    vl_last = out.vl_tokens[:, -1]
    assert torch.allclose(out.fed_inputs[:, 0], vl_last, atol=ATOL)


def test_parity_with_authors_latent_mode(tiny_model, processor, cfg):
    """Our batched, left-padded recurrence == authors' Qwen2_5_VLModel.forward(latent_mode=True)."""
    inp, _ = build_inputs(processor, cfg, "sft")
    assert inp.attention_mask[1].sum() < inp.attention_mask.shape[1]  # sample 1 is left padded
    st = _state_tokens(tiny_model)
    r = _reasoner(tiny_model, cfg)
    out = r(inp, st)
    emb = r.embed(inp, st)
    for b in range(2):
        fed, slot_out, prompt_h = authors_latent_mode_reference(tiny_model, inp, emb, b)
        assert fed.shape == (10, 64)
        torch.testing.assert_close(fed, out.fed_inputs[b], atol=ATOL, rtol=1e-4)
        torch.testing.assert_close(slot_out, out.latents[b], atol=ATOL, rtol=1e-4)
        valid = inp.attention_mask[b, :inp.slot_start].bool()
        torch.testing.assert_close(prompt_h, out.vl_tokens[b][valid], atol=ATOL, rtol=1e-4)


def test_parity_with_vllm_runner_state_machine(tiny_model, processor, cfg):
    inp, _ = build_inputs(processor, cfg, "vllm")
    st = _state_tokens(tiny_model)
    r = _reasoner(tiny_model, cfg)
    out = r(inp, st)
    emb = r.embed(inp, st)
    for b in range(2):
        fed, outs = vllm_runner_emulation(r, inp, emb, b, latent_size=10)
        torch.testing.assert_close(fed, out.fed_inputs[b], atol=ATOL, rtol=1e-4)
        torch.testing.assert_close(outs, out.latents[b], atol=ATOL, rtol=1e-4)


def test_conventions_differ(tiny_model, processor, cfg):
    """The authors' HF (sft) and vLLM conventions are not the same computation (audit finding)."""
    st = _state_tokens(tiny_model)
    a = _reasoner(tiny_model, cfg)(build_inputs(processor, cfg, "sft")[0], st).latents
    b = _reasoner(tiny_model, cfg)(build_inputs(processor, cfg, "vllm")[0], st).latents
    assert (a - b).abs().max() > 1e-3


@pytest.mark.parametrize("conv", ["sft", "vllm"])
def test_cached_equals_recompute_values_and_grads(tiny_model, processor, cfg, conv):
    inp, _ = build_inputs(processor, cfg, conv)
    st = _state_tokens(tiny_model).requires_grad_(True)
    res = {}
    for impl in ("cached", "recompute"):
        out = _reasoner(tiny_model, cfg, impl)(inp, st)
        loss = out.latents.pow(2).mean() + out.latents[:, -1].sum()
        (g,) = torch.autograd.grad(loss, st)
        res[impl] = (out.latents.detach(), g)
    torch.testing.assert_close(res["cached"][0], res["recompute"][0], atol=ATOL, rtol=1e-4)
    torch.testing.assert_close(res["cached"][1], res["recompute"][1], atol=1e-5, rtol=1e-3)


def test_not_a_soft_prompt_or_plain_hidden_extraction(tiny_model, processor, cfg):
    """Self-fed latents differ from (a) re-feeding a constant vector K times (soft prompt),
    (b) the D4 learned-query control, and (c) merely reading the last prompt hidden state."""
    inp, _ = build_inputs(processor, cfg, "sft")
    st = _state_tokens(tiny_model)
    r = _reasoner(tiny_model, cfg)
    out = r(inp, st)
    emb = r.embed(inp, st)
    P = inp.slot_start
    seq = torch.cat([emb[:, :P], out.fed_inputs[:, :1].expand(-1, 10, -1)], 1)
    pos = r.position_ids(inp)
    soft = r.lm(inputs_embeds=seq, position_ids=pos, attention_mask=inp.attention_mask).last_hidden_state[:, P:]
    assert torch.allclose(soft[:, 0], out.latents[:, 0], atol=ATOL)  # same first step by construction
    assert (soft[:, 1:] - out.latents[:, 1:]).abs().max() > 1e-3  # recurrence matters afterwards
    q = _reasoner(tiny_model, cfg, mode="learned_query")(inp, st)
    assert (q.latents - out.latents).abs().max() > 1e-3
    last = out.vl_tokens[:, -1:].expand(-1, 10, -1)
    assert (last[:, 1:] - out.latents[:, 1:]).abs().max() > 1e-3


def test_k0_vs_k10_share_context(tiny_model, processor, cfg):
    st = _state_tokens(tiny_model)
    inp0, _ = build_inputs(processor, cfg, "sft", K=0)
    inp10, _ = build_inputs(processor, cfg, "sft", K=10)
    o0 = _reasoner(tiny_model, cfg)(inp0, st)
    o10 = _reasoner(tiny_model, cfg)(inp10, st)
    assert o0.latents.shape == (2, 0, 64) and o10.latents.shape == (2, 10, 64)
    m = o0.vl_mask
    torch.testing.assert_close(o0.vl_tokens[m], o10.vl_tokens[m], atol=ATOL, rtol=1e-4)
    assert (o10.latents[:, -1] - o0.vl_tokens[:, -1]).abs().max() > 1e-3


def test_batch_vs_single_sample_parity(tiny_model, processor, cfg):
    """Left padding does not change any sample's latents (train/inference parity, B vs 1)."""
    inp, pb = build_inputs(processor, cfg, "sft")
    st = _state_tokens(tiny_model)
    r = _reasoner(tiny_model, cfg)
    both = r(inp, st).latents
    from conftest import INSTR, random_views

    views = random_views(2)
    for b in range(2):
        single = pb.build([views[b]], [INSTR[b]])
        one = r(single, st[b:b + 1]).latents
        torch.testing.assert_close(one[0], both[b], atol=1e-4, rtol=1e-4)


def test_train_mode_equals_eval_mode(tiny_model, processor, cfg):
    """Monet has attention_dropout=0: train()/eval() give identical latents under fixed inputs."""
    inp, _ = build_inputs(processor, cfg, "sft")
    st = _state_tokens(tiny_model)
    r = _reasoner(tiny_model, cfg)
    tiny_model.eval()
    a = r(inp, st).latents
    tiny_model.train()
    try:
        b = r(inp, st).latents
    finally:
        tiny_model.eval()
    torch.testing.assert_close(a, b, atol=1e-6, rtol=0)


def test_state_dependence_of_reasoning(tiny_model, processor, cfg):
    """Critical test 3: the latent states depend on proprioception injected *before* reasoning."""
    from monet_action.state_context import RobotStateProjector, embedding_rms

    inp, _ = build_inputs(processor, cfg, "sft")
    torch.manual_seed(0)
    proj = RobotStateProjector(8, 64, 2, 32, init_rms=embedding_rms(tiny_model.model.language_model.embed_tokens))
    r = _reasoner(tiny_model, cfg)
    s = torch.zeros(2, 8)
    a = r(inp, proj(s)).latents
    a2 = r(inp, proj(s.clone())).latents
    b = r(inp, proj(s + torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]))).latents
    torch.testing.assert_close(a, a2)
    assert (a - b).abs().max() > 1e-3
    # and every latent step (not only the first) is affected
    assert ((a - b).abs().amax(dim=(0, 2)) > 1e-4).all()
    # gradient of the last latent w.r.t. the raw state exists (state path is differentiable)
    s = s.requires_grad_(True)
    r(inp, proj(s)).latents[:, -1].sum().backward()
    assert s.grad is not None and s.grad.abs().sum() > 0


@torch.no_grad()
def test_greedy_text_decode_matches_full_recompute(tiny_model):
    """The Monet-7B GPU sanity-generation check uses this decoder (the authors' forward returns logits=None
    outside their SFT loss path, so HF ``generate()`` cannot be used)."""
    from monet_action.monet_loader import greedy_text_decode

    ids = torch.tensor([[151644, 872, 198, 3838, 374, 220, 17, 488, 220, 18, 30, 151645, 198, 151644, 77091, 198]])
    assert tiny_model(input_ids=ids).logits is None  # why generate() fails on the real checkpoint
    new = greedy_text_decode(tiny_model, ids, max_new_tokens=6)
    seq = ids
    ref = []
    for _ in range(6):  # no KV cache: full recompute each step
        h = tiny_model.model.language_model(input_ids=seq, use_cache=False, return_dict=True).last_hidden_state
        nxt = tiny_model.lm_head(h[:, -1]).argmax(-1, keepdim=True)
        ref.append(int(nxt))
        seq = torch.cat([seq, nxt], 1)
    assert new == ref and len(new) == 6
    assert greedy_text_decode(tiny_model, ids, max_new_tokens=6, stop_ids={ref[0]}) == ref[:1]
