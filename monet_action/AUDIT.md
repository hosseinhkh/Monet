# G0 audit and compatibility report

Status of evidence: everything below was read from source or measured in this repository's CPU container
(no GPU). Numbers marked *tiny* come from Monet's real modified code with a tiny random configuration; they
validate mechanisms, not Monet-7B. The Monet-7B rows of Gate 0 must be produced on the GPU server with
`bash scripts/audit_upstream.sh` (writes `results/audit/g0_report_monet7b.{json,md}`).

## 1. Monet: what "latent reasoning" actually is in the released code

Files: `monet_qwen_model/modeling_qwen2_5_vl_monet.py` (HF, used for SFT), `inference/vllm/monet_gpu_model_runner.py`
(vLLM, used for inference/evaluation), `RL/verl/workers/actor/dp_actor.py` + `RL/monet_models/*` (RL).

* **Mechanism.** In `Qwen2_5_VLModel.forward(latent_mode=True)` (section 2.2.2) each `<abs_vis_token_pad>`
  (id 151665) position receives as input embedding the **final, post-RMSNorm hidden state of the previous
  position** (`batch_last_hidden_state[b, pos-1]`), processed one token at a time on top of the KV cache.
  The vLLM runner does the same with `sample_hidden_states` (also post-norm) for `LATENT_SIZE` steps after
  the model samples `<abs_vis_token>` (151666), then emits `</abs_vis_token>` (151667).
* **Not `generate()`.** Stock `generate()` / ordinary final-hidden-state pooling never feeds hidden states back;
  it is not equivalent and is not used anywhere in this package.
* **Gradient in the authors' SFT path is truncated.** The step input is `latent_embed.detach()`; gradients
  reach the latent vectors only through a second "ce_patch" full forward. Our default `feedback_grad=bptt`
  keeps the whole K-step loop differentiable (no detach); `feedback_grad=truncated` reproduces the
  authors' one-step behaviour. Tested: `test_bptt_vs_truncated_feedback`.
* **Two different conventions in the authors' code (finding).** HF/SFT: `<abs_vis_token>` keeps its token
  embedding and the K pad slots after it are fed back. vLLM runner: the embedding of the sampled
  `<abs_vis_token>` itself is overwritten with the previous hidden state, followed by K-1 more fed slots
  (`rows = query_start_loc[1:]-1` selects the newly sampled token). The RL actor then patches latents into
  positions *strictly inside* (start, end), i.e. K-1 positions for K recorded latents, and skips samples
  whose counts differ. Both conventions are implemented (`monet.latent_convention: sft | vllm`); `sft` is
  the default because it is the differentiable training path. *tiny*: the two conventions give clearly
  different latents (cosine 0.26-0.28 per step, `g0_report_tiny.json`); measure on Monet-7B in G0.
* **Batch size.** The authors' `latent_mode` loops over samples ("only batch_size=1 tested"). Our
  implementation is batched with left padding so latent slots align across the batch.
* **Parity (tiny, CPU, fp32).** Batched cached recurrence vs the authors' `latent_mode` forward: max abs err
  7.2e-7 on fed vectors and slot outputs, including a left-padded sample; vs an emulation of the vLLM runner
  state machine: 7.2e-7; cached vs recompute implementation: 4.8e-7 (values) and gradients equal; self-feed
  identity `fed[k+1] == latent[k]` exact; the K=0 and K=10 prompts give identical VL context (err 0.0).
  The real vLLM runner itself cannot run without a GPU and is never used for backpropagation.
* **Checkpoint.** Monet-7B: hidden 3584, 28 layers, 28 heads / 4 KV heads, vision 32 blocks, vocab 151670,
  `latent_token_id=151665, latent_start_id=151666, latent_end_id=151667, answer_start_pattern=[151644, 77091]`.
  Saved with transformers 4.51.3 key names (`model.layers.*`, `visual.*`); the authors' class remaps them
  under 4.54.0. `load_monet` refuses to continue if any weight is missing.
* **Prompt format chosen for actions.** Qwen chat template; user turn = 2 image blocks (agent, wrist; 224x224
  -> 64 visual tokens each) + `Robot state: <|fim_pad|><|fim_pad|>` + task text; assistant turn starts with
  `<abs_vis_token>` followed by the K latent slots. The two `<|fim_pad|>` embeddings are replaced by the
  trained `RobotStateProjector` output, so the state is visible **before** the first latent step
  (`test_state_dependence_of_reasoning`: every latent step changes when the state changes).

## 2. GR00T N1.7 action head: reused vs re-implemented

GR00T N1.7 (`7aa0bca`) pairs a Cosmos-Reason2-2B (Qwen3-VL, width 2048) backbone with `Gr00tN1d7ActionHead`.
Its pretrained action weights are **not** compatible with Monet: cross-attention width 2048 (ours 512 from a
3584-wide backbone), 132-D padded state/action, horizon 40, 32 embodiments, 16 blocks x 32 heads x 48. We
therefore re-implement the contract at a compact size and train from scratch.

| GR00T component | Phase-1 treatment |
|---|---|
| Flow-matching math in `forward()`: `t = (1-Beta(1.5,1))*0.999`, `x_t=(1-t)eps+t a`, `v*=a-eps`, `t_disc=floor(1000 t)`, masked MSE `sum/(sum+1e-6)` | re-implemented identically (`flow_matching.py`, tested) |
| `get_action_with_features()` Euler loop, `t_i=i/N`, `dt=1/N`, default N=4 | re-implemented; N in {2,4,8}; RTC inpainting not used |
| `DiT` / `BasicTransformerBlock` (AdaLayerNorm(t), cross-attn on even blocks, self-attn on odd, GELU-tanh FFN, output AdaLN shift/scale) | re-implemented in pure PyTorch (no diffusers), 6 blocks x 512 x 8 heads |
| `TimestepEncoder` (diffusers sinusoid 256, flip_sin_to_cos, shift 1 -> MLP) | re-implemented; formula unit-tested |
| `MultiEmbodimentActionEncoder` (W1, sinusoidal tau, W2 swish, W3) | re-implemented for one embodiment |
| `CategorySpecificMLP` state encoder / action decoder | plain 2-layer MLPs (one embodiment) |
| learned position embedding on action tokens; `[state, actions]` sequence; decode last H tokens | kept |
| `strict_vl_padding_mask` | always on, plus a learned always-valid NULL key (no empty attention rows) |
| `vlln` + `vl_self_attention` | replaced by `LatentToActionConditioner` (separate LayerNorm+Linear for VL and latent tokens, type + latent-order embeddings, feature-source masks) |
| `AlternateVLDiT` (alternate image/text keys) | not used in Phase 1 |
| `state_dropout_prob=0.8`, 132-D padding, percentiles/relative-action options, augmentations | not copied; explicit LIBERO choices (no state dropout, 7/8-D, q01/q99 action norm, z-scored state) |
| Pretrained N1.7 action-head weights | not loaded |

## 3. LIBERO / data conventions (measured)

| Item | Finding | Evidence |
|---|---|---|
| Suite | libero_spatial, 10 tasks, 50 fixed init states each, 220-step cap | `libero_task_table()`, `LiberoEnv` |
| Controller | OSC_POSE, 20 Hz, input range +-1, output max 0.05 m / 0.5 rad per step | `env.env.robots[0].controller` |
| Action | 7D: delta-EEF xyz (base frame), delta axis-angle, gripper | data |
| Gripper | -1 = open, +1 = close (binary in demos); finger q decreases after +1 | `test_gripper_and_action_conventions` |
| Translation scale | corr(action, delta eef) = 0.94/0.98/0.94, about 11-13 mm per unit action per frame | `g0_report_tiny.json` (data section) |
| Frequency | one dataset frame == one `env.step` (5 sim steps of +1 x moved 44 mm from rest; consistent with ~11 mm/frame in data). Dataset metadata says `fps: 10`; treat it as metadata only | audit `sim` section |
| State (8D) | eef_pos(3) + quat2axisangle(eef_quat)(3) + gripper qpos(2); axis-angle norm about pi. The dataset feature *names* ("rx, ry, rz, rw, gripper") are wrong | data + sim state match after reset |
| Images | 256x256 agent + wrist; simulator frames must be rotated 180 deg (`[::-1, ::-1]`) to match the demos: corr 0.92 (agent) / 0.91 (wrist) vs -0.26 unrotated | audit `sim` section |
| Task ids | LeRobot `task_index` order differs from LIBERO's task order: mapped by instruction text (e.g. LeRobot 0 -> LIBERO 6) | `libero_task_table()` |
| Episodes | 432 successful demos (failed replays and no-op actions removed upstream); frames contiguous per episode; windows never cross episodes (last action repeated, loss-masked) | `test_no_cross_episode_leakage_and_horizon_padding` |
| Reset | `env.reset()` -> `set_init_state(init_states[i])` -> 10 no-op steps `[0,0,0,0,0,0,-1]` (not counted) | `LiberoEnv.reset` |
| Init-state files | numpy pickles; torch>=2.6 needs `weights_only=False` (trusted, pinned upstream files) | `LiberoEnv.__init__` |
| Privileged data | `PolicyObservation` has only images, state, instruction and the agent's own previously executed actions (TOOL_RESULT); success flags live in `diagnostics` | `test_policy_observation_has_no_oracle_fields` |
| End-to-end action convention | raw LIBERO demos (hdf5 with recorded MuJoCo states) replayed in our robosuite 1.4.1 / MuJoCo 3.3.2 build solve the task: demos 0-3 of task 0 succeed at steps 82/73/77/79; our 8D state matches the recorded `ee_states` (within 3 mm and 0.01 rad on demos 0-3, measured right after `set_init_state`) | `test_recorded_demo_replays_to_success`, `test_state_conversion_matches_raw_demo` |
| Reset rendering (finding + fix) | LIBERO hard-resets (reloads the scene XML) on every `reset()` and re-samples fixture textures (stove, cabinet) from the *global* `random`/`numpy` RNGs: the same init state gave bit-identical MuJoCo state but 12-16% different pixels across resets, i.e. "matched" episodes were not visually matched across variants. `LiberoEnv.reset` now seeds those RNGs from (task, init state) and restores them; observations are pixel-identical | `test_reset_is_deterministic_and_init_states_differ` |
| Final vs development init states | final reporting uses benchmark init states 0..29 (3 seeds x 10); smoke rollouts, in-training sim checks and the D3 swap bank use 30..49 only | `test_final_and_development_init_states_are_disjoint` |

## 4. Repository findings

* The upstream root `.gitignore` ignores every directory named `scripts/` (and `logs/`). The first push
  of this branch therefore **silently omitted `monet_action/scripts/`**. Fixed by re-including the
  directory in `monet_action/.gitignore`; `tests/test_repo_hygiene.py` now fails if an essential file
  is ignored.
* Upstream junk removed: `transformers/.DS_Store`, the only file in that directory. The full upstream
  review is in `docs/UPSTREAM_CODE_REVIEW.md`.

## 5. Gate status

| Gate item | Status |
|---|---|
| Source SHAs pinned | done (`SOURCE_VERSIONS.md`) |
| Dependency lock | CPU lock done; server lock written by `scripts/setup_env.sh` |
| Monet latent mechanism reproduced, parity vs authors' code | done on tiny config (CPU); **pending on Monet-7B (GPU)** |
| K=0 vs K=10, shapes, finite values, state dependence | done on tiny config; **pending on Monet-7B** |
| One correct LIBERO batch | done (real data; `test_one_real_minibatch`) |
| Simulator reset/step/reobserve + orientation | done (real LIBERO, OSMesa) |
| GR00T compatibility report | done (section 2) |
