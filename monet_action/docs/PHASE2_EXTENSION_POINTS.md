# Phase-2 extension points (CoVT perceptual thought tokens) — not implemented in Phase 1

Phase 1 deliberately trains no perception tokens. The interfaces that Phase 2 would extend:

* **Prompt (`prompt.py`).** Perceptual-thought tokens would be placed *before* `<abs_vis_token>`, the
  same way the state (`<|fim_pad|>`) and TOOL_RESULT (`<|fim_middle|>`) tokens are. That puts them in the
  KV cache the self-fed latent steps attend to. Add a new placeholder token and a projector, as
  `RobotStateProjector` does.
* **Backbone (`MonetLatentReasoner.embed`).** `_scatter(emb, mask, tokens, name)` is the single place
  where projected tokens enter Monet's embedding stream.
* **Conditioner (`conditioning.py`).** Token groups are typed (`TOKEN_NULL / TOKEN_VL / TOKEN_LATENT`) and
  masked separately. A `TOKEN_PERCEPT` group with its own LayerNorm + projection would keep the same DiT
  (`CALL_ACTION`) contract `[B, N, C] + mask`.
* **Diagnostics.** The `feature_source` switch and the D3 interventions generalise to a new group:
  mask it, shuffle it, swap it.
* **Teacher features (DINO, depth, SAM).** These belong to training-time losses only and must never be a
  policy input at evaluation. `PolicyObservation` enforces the allowed inputs, and a test checks it.
