"""Differentiable Monet latent reasoning (genuine self-fed hidden-state recurrence).

Source of truth: ``Qwen2_5_VLModel.forward(latent_mode=True)`` in the authors' modeling file
(monet_qwen_model/modeling_qwen2_5_vl_monet.py, section 2.2.2) and the vLLM runner
(inference/vllm/monet_gpu_model_runner.py, "ABS-VIS" blocks). Monet's latent step is:

    input_embedding[slot_j] := last_hidden_state[slot_j - 1]      (final layer, post-RMSNorm)

i.e. the hidden state produced at the previous position is fed back as the next input
embedding, autoregressively, on top of the KV cache of the multimodal prompt. We re-implement
exactly this, batched (left padding keeps slots aligned), with three differences that are all
configurable and covered by tests:

* ``feedback_grad="bptt"`` (default) keeps the fed-back vector in the autograd graph, so the
  action loss back-propagates through all K steps. The authors' SFT ``latent_mode`` detaches
  the step input (``inputs_embeds=latent_embed.detach()``) and recovers gradients through a
  second "ce_patch" pass; ``feedback_grad="truncated"`` reproduces that one-step gradient.
* ``recurrence_impl="recompute"`` replaces the KV-cache loop by K+1 full forwards on the
  growing sequence. Same function (parity-tested), but compatible with HF gradient
  checkpointing, which disables the cache in training mode.
* ``reasoning_mode="learned_query"`` is the D4 matched-compute control: the K slots receive
  trainable query embeddings instead of fed-back hidden states (same prompt, same slots, same
  number of transformer token passes, no self-feedback).

Standard ``generate()`` / plain hidden-state pooling is *not* Monet latent reasoning; this
module never falls back to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from .monet_loader import IMAGE_TOKEN_ID
from .profiling import Timer
from .prompt import MonetInputs


@dataclass
class BackboneOutput:
    vl_tokens: torch.Tensor  # [B, P, D]  prompt hidden states (state-aware context)
    vl_mask: torch.Tensor  # [B, P] bool
    latents: torch.Tensor  # [B, K, D]  hidden states produced at the K latent slots
    fed_inputs: torch.Tensor  # [B, K, D]  vectors actually fed into the K slots
    timings: Dict[str, float] = field(default_factory=dict)


def _select_layer(out, layer: int) -> torch.Tensor:
    if layer == -1:
        return out.last_hidden_state
    if out.hidden_states is None:
        raise RuntimeError("hidden states were not requested")
    return out.hidden_states[layer]


class MonetLatentReasoner(nn.Module):
    """Wraps a (possibly LoRA-adapted) Monet ``Qwen2_5_VLForConditionalGeneration``."""

    def __init__(self, model: nn.Module, monet_cfg, conditioner_cfg=None):
        super().__init__()
        self.model = model
        self.cfg = monet_cfg
        self.cond_cfg = conditioner_cfg
        D = self.hidden_size
        self._timer = Timer(None)
        self.learned_queries = None
        if monet_cfg.reasoning_mode == "learned_query":
            from .state_context import embedding_rms

            rms = embedding_rms(self.embed_tokens)
            q = torch.randn(max(1, monet_cfg.num_latents), D) * rms
            self.learned_queries = nn.Parameter(q.to(torch.float32))

    # ------------------------------------------------------------------ accessors
    @property
    def core(self):
        return self.model.model  # Qwen2_5_VLModel (authors' class)

    @property
    def lm(self):
        return self.model.model.language_model

    @property
    def embed_tokens(self):
        return self.model.model.language_model.embed_tokens

    @property
    def hidden_size(self) -> int:
        return self.model.config.text_config.hidden_size

    @property
    def dtype(self):
        return self.embed_tokens.weight.dtype

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def _scatter(emb: torch.Tensor, mask: torch.Tensor, tokens: torch.Tensor, what: str) -> torch.Tensor:
        n = int(mask[0].sum())
        if tokens.shape[1] != n or not bool((mask.sum(1) == n).all()):
            raise RuntimeError(f"{what} token / placeholder count mismatch")
        return emb.masked_scatter(mask.unsqueeze(-1).expand_as(emb), tokens.to(emb.dtype).reshape(-1, emb.shape[-1]))

    def embed(self, inputs: MonetInputs, state_tokens: Optional[torch.Tensor],
              tool_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embed_tokens(inputs.input_ids)
        if inputs.pixel_values is not None:
            ctx = torch.no_grad() if self.cfg.freeze_vision else torch.enable_grad()
            with ctx:
                img = self.core.get_image_features(inputs.pixel_values.to(self.dtype), inputs.image_grid_thw)
                img = torch.cat(list(img), 0) if isinstance(img, (list, tuple)) else img
            mask = (inputs.input_ids == IMAGE_TOKEN_ID).unsqueeze(-1).expand_as(emb)
            if int(mask[..., 0].sum()) != img.shape[0]:
                raise RuntimeError("image features / image tokens mismatch")
            emb = emb.masked_scatter(mask, img.to(emb.dtype))
        if state_tokens is not None:
            emb = self._scatter(emb, inputs.state_mask, state_tokens, "state")
        if inputs.tool_mask is not None:
            if tool_tokens is None:
                raise RuntimeError("prompt has TOOL_RESULT placeholders but no tool tokens were given")
            emb = self._scatter(emb, inputs.tool_mask, tool_tokens, "TOOL_RESULT")
        return emb

    def position_ids(self, inputs: MonetInputs) -> torch.Tensor:
        pos, _ = self.core.get_rope_index(inputs.input_ids, inputs.image_grid_thw, None,
                                          attention_mask=inputs.attention_mask)
        return pos  # [3, B, L]

    def _vl_mask(self, inputs: MonetInputs, P: int) -> torch.Tensor:
        m = inputs.attention_mask[:, :P].bool()
        sel = getattr(self.cond_cfg, "vl_token_select", "all") if self.cond_cfg is not None else "all"
        if sel == "image_only":
            m = m & (inputs.input_ids[:, :P] == IMAGE_TOKEN_ID)
        elif sel == "text_only":
            m = m & (inputs.input_ids[:, :P] != IMAGE_TOKEN_ID)
        return m

    # ------------------------------------------------------------------ forward
    def forward(self, inputs: MonetInputs, state_tokens: Optional[torch.Tensor] = None,
                num_latents: Optional[int] = None, tool_tokens: Optional[torch.Tensor] = None,
                timer: Optional[Timer] = None) -> BackboneOutput:
        """Timings (ms, disjoint): monet_embed (vision tower + token embedding + RoPE ids),
        monet_prefill (prompt forward) and monet_latent_steps (the K self-fed steps)."""
        K = inputs.num_latents if num_latents is None else num_latents
        if K != inputs.num_latents:
            raise ValueError("num_latents must match the prompt that was built")
        own_timer = timer is None
        timer = timer or Timer(inputs.input_ids.device)
        self._timer = timer
        with timer("monet_embed"):
            emb = self.embed(inputs, state_tokens, tool_tokens)
            pos = self.position_ids(inputs)
        am = inputs.attention_mask
        P = inputs.slot_start
        need_hidden = self.cfg.latent_layer != -1 or self.cfg.vl_layer != -1
        if self.cfg.reasoning_mode == "learned_query":
            vl, lat, fed = self._learned_query(emb, pos, am, P, K, need_hidden)
        elif self.cfg.recurrence_impl == "cached":
            vl, lat, fed = self._cached(emb, pos, am, P, K, need_hidden)
        else:
            vl, lat, fed = self._recompute(emb, pos, am, P, K, need_hidden)
        vl_mask = self._vl_mask(inputs, P)
        # padded / unselected positions never reach the action head (also guards against
        # non-finite values at fully-masked padded query rows under some attention kernels)
        vl = vl.masked_fill(~vl_mask.unsqueeze(-1), 0.0)
        if self.cond_cfg is not None and vl.shape[1] > self.cond_cfg.vl_max_tokens:
            n = self.cond_cfg.vl_max_tokens
            vl, vl_mask = vl[:, -n:], vl_mask[:, -n:]
        return BackboneOutput(vl_tokens=vl, vl_mask=vl_mask, latents=lat, fed_inputs=fed,
                              timings=timer.result() if own_timer else {})

    # -------------------------------------------------------------- KV-cache impl
    def _lm(self, embeds, pos, am, cache, cache_position, need_hidden):
        return self.lm(input_ids=None, inputs_embeds=embeds, position_ids=pos, attention_mask=am,
                       past_key_values=cache, use_cache=cache is not None, cache_position=cache_position,
                       output_hidden_states=need_hidden, return_dict=True)

    def _feed(self, h: torch.Tensor) -> torch.Tensor:
        return h.detach() if self.cfg.feedback_grad == "truncated" else h

    def _cached(self, emb, pos, am, P, K, need_hidden):
        dev = emb.device
        with self._timer("monet_prefill"):
            cache = DynamicCache()
            out = self._lm(emb[:, :P], pos[:, :, :P], am[:, :P], cache, torch.arange(P, device=dev), need_hidden)
            vl = _select_layer(out, self.cfg.vl_layer)
            cache = out.past_key_values
            h = out.last_hidden_state[:, -1]
        lats, feds = [], []
        with self._timer("monet_latent_steps"):
            for j in range(K):
                p = P + j
                x = self._feed(h)
                feds.append(x)
                o = self._lm(x.unsqueeze(1), pos[:, :, p:p + 1], am[:, :p + 1], cache,
                             torch.tensor([p], device=dev), need_hidden)
                cache = o.past_key_values
                h = o.last_hidden_state[:, -1]
                lats.append(_select_layer(o, self.cfg.latent_layer)[:, -1])
        return vl, self._stack(lats, emb), self._stack(feds, emb)

    # -------------------------------------------------------------- recompute impl
    def _recompute(self, emb, pos, am, P, K, need_hidden):
        cur = emb[:, :P]
        feds = []
        vl = None
        for j in range(K + 1):
            L = P + j
            with self._timer("monet_prefill" if j == 0 else "monet_latent_steps"):
                o = self._lm(cur, pos[:, :, :L], am[:, :L], None, None, need_hidden)
            if j == 0:
                vl = _select_layer(o, self.cfg.vl_layer)
            if j < K:
                x = self._feed(o.last_hidden_state[:, -1])
                feds.append(x)
                cur = torch.cat([cur, x.unsqueeze(1)], 1)
            else:
                lat = _select_layer(o, self.cfg.latent_layer)[:, P:P + K]
        if K == 0:
            lat = emb.new_zeros(emb.shape[0], 0, emb.shape[-1])
        return vl, lat, self._stack(feds, emb)

    # -------------------------------------------------------------- D4 control
    def _learned_query(self, emb, pos, am, P, K, need_hidden):
        dev = emb.device
        B = emb.shape[0]
        q = self.learned_queries[:K].to(emb.dtype).unsqueeze(0).expand(B, -1, -1)
        if self.cfg.learned_query_parallel or K == 0:
            seq = torch.cat([emb[:, :P], q], 1)
            with self._timer("monet_prefill"):  # prompt + K query slots in one forward
                o = self._lm(seq, pos[:, :, :P + K], am[:, :P + K], None, None, need_hidden)
            vl = _select_layer(o, self.cfg.vl_layer)[:, :P]
            lat = _select_layer(o, self.cfg.latent_layer)[:, P:P + K]
            return vl, lat, q
        with self._timer("monet_prefill"):
            cache = DynamicCache()
            out = self._lm(emb[:, :P], pos[:, :, :P], am[:, :P], cache, torch.arange(P, device=dev), need_hidden)
            vl, cache = _select_layer(out, self.cfg.vl_layer), out.past_key_values
        lats = []
        with self._timer("monet_latent_steps"):
            for j in range(K):
                p = P + j
                o = self._lm(q[:, j:j + 1], pos[:, :, p:p + 1], am[:, :p + 1], cache,
                             torch.tensor([p], device=dev), need_hidden)
                cache = o.past_key_values
                lats.append(_select_layer(o, self.cfg.latent_layer)[:, -1])
        return vl, self._stack(lats, emb), q

    @staticmethod
    def _stack(xs, ref):
        if not xs:
            return ref.new_zeros(ref.shape[0], 0, ref.shape[-1])
        return torch.stack(xs, 1)


# ======================================================================================
# Reference implementations used only for parity tests / audit (never for training)
# ======================================================================================
@torch.no_grad()
def authors_latent_mode_reference(model, inputs: MonetInputs, inputs_embeds: torch.Tensor, b: int):
    """Run the authors' ``Qwen2_5_VLModel.forward(latent_mode=True)`` on sample ``b``.

    Only valid for the "sft" convention (that function looks for <abs_vis_token_pad> ids).
    The authors only support batch size 1, so left padding is stripped first.
    Returns (fed_vectors [K, D], slot_outputs [K, D], prompt_hidden [P_b, D]).
    """
    if inputs.convention != "sft":
        raise ValueError("authors' HF latent_mode implements the sft convention")
    valid = inputs.attention_mask[b].bool()
    ids = inputs.input_ids[b][valid].unsqueeze(0)
    am = torch.ones_like(ids)
    emb = inputs_embeds[b][valid].unsqueeze(0)
    n_views = None
    pv, thw = None, None
    if inputs.image_grid_thw is not None:
        n_views = inputs.image_grid_thw.shape[0] // inputs.batch_size
        thw = inputs.image_grid_thw[b * n_views:(b + 1) * n_views]
        counts = inputs.image_grid_thw.prod(-1)
        start = int(counts[:b * n_views].sum())
        pv = inputs.pixel_values[start:start + int(thw.prod(-1).sum())]
        # The authors' forward scatters image features itself; give it the raw token embeddings
        # at image positions (they will be overwritten) and keep any state tokens we injected.
    core = model.model
    core.rope_deltas = None
    out = core(input_ids=ids, attention_mask=am, inputs_embeds=emb, pixel_values=pv, image_grid_thw=thw,
               latent_mode=True, use_cache=True, return_dict=True)
    lat_pos = (ids[0] == model.config.latent_token_id).nonzero().flatten()
    fed = out.ce_patch_vec[0]
    slot_out = out.last_hidden_state[0, lat_pos]
    first = int(lat_pos[0]) if len(lat_pos) else ids.shape[1]
    return fed, slot_out, out.last_hidden_state[0, :first]


@torch.no_grad()
def vllm_runner_emulation(reasoner: MonetLatentReasoner, inputs: MonetInputs, inputs_embeds: torch.Tensor,
                          b: int, latent_size: int):
    """Token-by-token emulation of the authors' vLLM ``GPUModelRunner`` ABS-VIS state machine.

    Mirrors inference/vllm/monet_gpu_model_runner.py: when ``<abs_vis_token>`` is *sampled*,
    ``pending`` := hidden state of the current last position and the runner turns ``active``;
    at every following decode step the embedding of the newly scheduled token is overwritten
    with ``pending`` and ``pending`` is refreshed from that step's last hidden state, until
    ``current_len >= LATENT_SIZE`` (then ``</abs_vis_token>`` is emitted). Here the prompt
    *before* ``<abs_vis_token>`` is prefilled and ``<abs_vis_token>`` is forced as the sampled
    token. Returns (pending vectors fed [K, D], hidden states at the overwritten slots [K, D]).
    """
    valid = inputs.attention_mask[b].bool()
    ids = inputs.input_ids[b][valid].unsqueeze(0)
    emb = inputs_embeds[b][valid].unsqueeze(0)
    am_full = torch.ones_like(ids)
    sub = MonetInputs(ids, am_full, None, None, inputs.state_mask[b][valid].unsqueeze(0), inputs.num_latents,
                      0, inputs.convention)
    if inputs.image_grid_thw is not None:
        n_views = inputs.image_grid_thw.shape[0] // inputs.batch_size
        sub.image_grid_thw = inputs.image_grid_thw[b * n_views:(b + 1) * n_views]
    pos = reasoner.position_ids(sub)
    start_idx = int((ids[0] == 151666).nonzero()[0])  # position of <abs_vis_token>
    lm = reasoner.lm
    cache = DynamicCache()
    out = lm(inputs_embeds=emb[:, :start_idx], position_ids=pos[:, :, :start_idx], attention_mask=am_full[:, :start_idx],
             past_key_values=cache, use_cache=True, cache_position=torch.arange(start_idx, device=ids.device))
    cache = out.past_key_values
    st = {"active": True, "pending": out.last_hidden_state[0, -1], "current_len": 1}  # <abs_vis_token> sampled
    fed, outs = [], []
    p = start_idx
    while True:
        x = st["pending"]  # override of the newly scheduled token's embedding
        st["pending"] = None
        fed.append(x)
        o = lm(inputs_embeds=x.view(1, 1, -1), position_ids=pos[:, :, p:p + 1], attention_mask=am_full[:, :p + 1],
               past_key_values=cache, use_cache=True, cache_position=torch.tensor([p], device=ids.device))
        cache = o.past_key_values
        h = o.last_hidden_state[0, -1]
        outs.append(h)
        p += 1
        if st["current_len"] >= latent_size:  # sampled token replaced by </abs_vis_token>
            break
        st["pending"] = h
        st["current_len"] += 1
    return torch.stack(fed), torch.stack(outs)
