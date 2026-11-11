# Research plan: Monet-Action v2, Phase 1, and how this repository implements it

Source: *Monet-Action v2 — Continuous latent reasoning with a callable flow-matching DiT action expert*,
research proposal, revision 2 (9 November 2026). Phase 1: Monet-7B + action tool + LIBERO-Spatial, on one
~96 GB NVIDIA Blackwell GPU. **No training results are claimed.**

## 1. The idea

Vision-language-action models usually condition action generation on visual and language features. Monet-7B
adds something unusual: **continuous, self-fed latent reasoning**. After reading the prompt, it runs K extra
steps in which each step's input is the final hidden state of the previous step (no words, no image tokens).
GR00T N1.7 provides a proven design for a **flow-matching Diffusion Transformer (DiT) action expert**
conditioned on vision-language features and robot state.

Phase 1 joins the two. Monet's latent reasoning states become the context of a **callable action tool**: a
compact GR00T-style DiT that turns them into continuous robot actions.

* **Research question.** Do Monet's self-fed continuous latent reasoning states improve closed-loop robotic
  manipulation when they condition a learned continuous-action expert, relative to a matched model without
  those reasoning steps?
* **Cost question.** Does any improvement offset the extra GPU time and action latency?
* **Novelty hypothesis (not a result).** Continuous internal reasoning used as an *executable action-tool
  context* can improve sample efficiency, task success, or robustness under a controlled compute budget.
  Simply coupling a VLM to a DiT is not a novelty claim. Related latent-VLA work: LaST0 and LaRA-VLA.

**Executive decisions in the proposal:**

* The primary action head is a compact GR00T-style flow-matching DiT, trained from scratch. An MLP head is
  kept only as an engineering and scientific control.
* Do **not** insert the full GR00T model or assume its pretrained action weights fit Monet.
* No CoVT in Phase 1.

## 2. Architecture

```
LIBERO agent + wrist RGB + instruction + 8D proprioception + TOOL_RESULT (previously executed action)
  -> Monet-7B (customised Qwen2.5-VL forward); trainable state context injected BEFORE reasoning
  -> K self-fed continuous latent steps (default K=10)                  latents [B, K, 3584]
  -> conditioning adapter: VL tokens + ordered latents, typed, masked   [B, N, 512]
  -> CALL_ACTION (fixed dispatch)
  -> compact flow-matching DiT + robot-state encoder + action encoder   normalised chunk [B, H=8, A=7]
  -> execute E=1 or 2 actions in LIBERO -> new RGB + state + applied action = TOOL_RESULT -> next decision
```

* **`CALL_ACTION` is a fixed runtime action in Phase 1.** Monet does not learn to choose among tools; a
  learned OBSERVE / THINK / PLAN / ACT router is Phase 3.
* **Where gradients flow.** The action head is differentiable. The environment is not differentiated:
  no backpropagation through MuJoCo.
* **No `generate()`.** Monet's latent inference uses a customised forward / vLLM path, and standard
  `generate()` is not equivalent to it. So the training path is a differentiable PyTorch reimplementation,
  parity-tested against the authors' code.

**Flow matching.** With a normalised action chunk `a`, noise `eps ~ N(0, I)` and time `t` in [0, 1]:

* `x_t = (1 - t)·eps + t·a`
* target velocity `v* = a - eps`
* masked loss `L_FM = E[ ||M ⊙ (v_θ(x_t, t, c, s) - v*)||² / (ΣM + δ) ]`

At inference the velocity field is integrated from noise in 4 Euler steps; 2 and 8 are ablated.

**Differentiability rule.** `L_FM` must update the DiT and the adapter. In joint training it must also give
finite, non-zero gradients to the Monet LoRA parameters *through the genuine K-step latent recurrence*.

## 3. Hypotheses and how each is tested

| hypothesis | test in this repository | where reported |
|---|---|---|
| **H1 Feasibility**: Monet gives a differentiable, useful conditioning interface to a flow-matching DiT | gradient audit at the start of every run; G1 overfit; D0/D1/D2 reach non-trivial success | `gradient_audit.json`, preflight `smoke`, report §1 |
| **H2 Added value**: genuine self-fed K-step recurrence improves held-out closed-loop success or sample efficiency vs matched controls | primary contrasts D1 vs D0 (frozen) and D2 vs D0-LoRA (joint); D1 vs D4 (self-feedback vs extra compute) | report §1–2, `RESEARCH_MEMO.md` |
| **H3 Dependence**: structured perturbations of the latents change action quality consistently with a causal path (OOD caveat) | D3 interventions on D2: zero, gaussian, swap, shuffle order, truncate, mask; bypass checks `mask_vl`, `mask_all` | report §4 |
| **H4 Cost**: any gain offsets the added reasoning and denoising time | latency p50/p95 split into Monet / DiT / preprocessing; GPU-seconds per success; K sweep; Euler steps 2/4/8 | report §4b, §8, `latency.json` |

All four answers count: positive, null and negative results each answer a real question. A paper-level claim
would need gains not explained by extra capacity or compute, or a clear quality–cost advantage.

## 4. Variants

| id | Monet | what it isolates | config |
|---|---|---|---|
| D0 | frozen, K=0 | primary no-reasoning control | `d0_k0.yaml` |
| D1 | frozen, K=10 | are released Monet latents useful without updating Monet? | `d1_k10.yaml` |
| D2 | LoRA, K=10 | primary jointly trained model; `L_FM` flows through the recurrence into LoRA | `d2_lora.yaml` |
| D3 | D2 at evaluation | dependence diagnostic (not proof of interpretable reasoning) | `--intervention …` |
| D4 | frozen, K=10 learned queries | same extra computation without self-feedback ("strongly recommended") | `d4_query.yaml` |
| D5 | frozen, K=10, MLP head | engineering control only | `d5_mlp.yaml` |

**Added in this repository** (all documented, none replaces a proposal variant):

* **D0-LoRA** (`d0_k0_lora.yaml`): D2's LoRA with K=0. D2 vs D0 changes both K and LoRA; D2 vs D0-LoRA
  changes only K.
* **D4-LoRA**: the D4 control under joint training.
* **K sweep** {5, 20}.
* **Ablations**: latents-only conditioning, no previous action in TOOL_RESULT, the SFT-stage latent layout.
* **Monet adaptation capacity ladder**: wide LoRA, full fine-tuning of 4 or 8 decoder layers, each with a
  matched K=0 control. It asks whether D2's small LoRA limits the result. See [FINETUNING.md](FINETUNING.md).

Action-head size, training samples, seeds and evaluation conventions are matched across D0–D2 (D0 and D1
have identical parameter counts, which is tested). Because more latent steps cost more compute, results are
reported as a quality-versus-compute frontier, not only the highest success.

## 5. Work packages, gates and status

| WP / gate | proposal requirement | implementation | status |
|---|---|---|---|
| WP0 | pin Monet, GR00T, LIBERO, data commits; versions and licences | `SOURCE_VERSIONS.md`, `environment/`, `scripts/setup_env.sh` (isolated conda env) | done |
| WP0 | verify Monet-7B loading and K=10 latents; parity with the released inference | `monet_backbone.py` + parity emulations; `tests/test_monet7b_gpu.py` | passed on the GPU server |
| WP0 | inspect GR00T N1.7 action head, borrow the design | `AUDIT.md` §2 (reused vs re-implemented), `dit_action.py` | done |
| WP0 | LIBERO data: camera keys, proprio/action schema, segmentation, normalisation, execution conventions | `libero_dataset.py`, `audit.py` (`audit_data`, `audit_sim`) | passed on the GPU server |
| **Gate 0** | verified SHAs, lockfile, exact latent shape/value checks, one correct LIBERO minibatch | preflight `env`, `weights`, `tests_gpu`, `audit` | **passed** |
| WP1 | MonetLatentAdapter, RobotStateProjector, LatentToActionConditioner, FlowMatchingDiT, typed CALL_ACTION / TOOL_RESULT | `monet_backbone.py`, `state_context.py`, `conditioning.py`, `dit_action.py`, `tool_runtime.py` | done |
| WP1 | K=0 and K=10, state injected before reasoning | `prompt.py` (state placeholders before the latent slots) | done, tested |
| WP1 | 7D / 8-step batches with masks; stable normalisation inverse | `libero_dataset.py`, `normalization.py` | done, tested |
| WP1 | velocity loss, 4-step integration, other solver step counts | `flow_matching.py` | done, tested |
| WP1 | tests: shapes, NaN/Inf, latent perturbations, masks, gradient paths, a full reset/execute/reobserve cycle | `tests/` (190+ tests) | done |
| WP1 | overfit a tiny batch; tool runtime with non-trivial state changes | `test_train_smoke.py`, preflight `smoke` | passed on the GPU server |
| **Gate 1** | synthetic tests pass, one real minibatch trains, non-constant 7D commands, a closed-loop step works | preflight `tests_cpu`, `dry_run`, `smoke` | **passed** |
| WP2 | D0–D5 trained, matched | `configs/`, `train.py`, `scripts/train_all.sh` | ready; not yet run |
| WP3 | 2-task smoke, then all 10 tasks; 3 seeds × ≥10 episodes; matched init states; episode-disjoint validation | `evaluate.py`, `scripts/evaluate_all.sh` (init states 0–29 final, 30–49 development) | ready; not yet run |
| WP3 | per-task success + uncertainty; latency p50/p95 (observation→action, Monet, DiT); DiT steps; peak memory; GPU-seconds; action calls; simulator wall-clock; validation loss | `metrics.py`, `report.py` | ready |
| WP3 | GPU-seconds per successful episode; dollars per success if an hourly cost is given | `metrics.aggregate_episodes`, `eval.gpu_hourly_cost_usd` | ready |
| WP3 | latency for K=0/10 × solver 2/4/8; best operating point | `evaluate.py latency`, report §8 | ready |
| WP3 | success and failure videos; failure taxonomy (localisation, gripper timing, action frame, stalling, oscillation, latency) | `videos/`, `tool_runtime.auto_failure_tags` (hints) + hand labels | ready |
| **Gate 2** | reloadable D0/D1/D2 checkpoints, full task × seed matrix, D3 interventions, resource profiles; negative results stated | training + evaluation | pending (training) |

## 6. Risks and their mitigations in the code

| risk | mitigation |
|---|---|
| latent bypass | gradient audit; D3 replacement / ablation; conditioned vs unconditioned (`mask_vl`, `mask_all`) |
| DiT works offline but fails in control | coordinate frame, gripper mapping, normalisation, reset and timing audited (`audit_sim`, `test_libero_sim.py`); smoke rollouts early |
| Monet training / inference mismatch | the training forward is parity-tested against the authors' `latent_mode` forward and an emulation of their vLLM runner; never falls back to HF generation |
| GR00T incompatibility | only the action-head *design* is reused, in a smaller tested implementation; no GR00T weights |
| excessive latency | K-step reasoning and DiT integration are timed separately |
| GPU out-of-memory | bf16, LoRA, frozen vision, gradient accumulation, optional activation checkpointing; memory profiled before training |
| misleading gains | matched data, DiT, episodes and seeds; confidence intervals; compute-matched D4 |
| privileged information leakage | `PolicyObservation` has no simulator object state (tested) |
| over-interpreting reasoning | dependence is not interpretability; claims limited to the experiments |

## 7. Roadmap after Phase 1 (from the proposal)

* **Phase 2: CoVT-style internal perceptual thoughts.** Distil compact, internally generated visual-thought
  tokens (DINO, depth, optional SAM supervision) and place them *before* Monet's latent reasoning. Reuse the
  same DiT tool interface; compare 0 vs 8/16/20 perceptual tokens. Extension points:
  [PHASE2_EXTENSION_POINTS.md](PHASE2_EXTENSION_POINTS.md).
* **Phase 3: tool routing.** A small discrete THINK_AGAIN / OBSERVE / CALL_ACTION / CALL_PLANNER / STOP
  policy; fast action loop decoupled from slow reasoning; safety kept outside the model.
* **Phase 4: cost and latency.** Asynchronous execution, caching and selective regeneration of reasoning
  states, distillation; comparison with agentic systems such as ASPIRE.
* **Phase 5: long-horizon autonomy.** Longer tasks, event memory, geometric planning, recovery, skill reuse.

## 8. Deliverables (proposal §7)

| # | deliverable | status |
|---|---|---|
| 1 | reproducible source, pinned commits, isolated environment, data licences | done |
| 2 | verified Monet K-step forward with tests and a training/inference parity log | done; real-model parity passed on the GPU server |
| 3 | trainable compact DiT with exact training/inference contracts, plus the MLP sanity head | done |
| 4 | trained D0/D1/D2 checkpoints, D3/D4 evaluations, gradient audit, restart/resume scripts | scripts done; checkpoints pending |
| 5 | LIBERO-Spatial closed-loop results table, failure videos, latency/compute profile | pending (`results/phase1_report.md`) |
| 6 | short research conclusion: what is supported, what is not, whether Phase 2 is justified | pending (`RESEARCH_MEMO.md`) |

## References (from the proposal)

1. Wang et al. *Monet: Reasoning in Latent Visual Space Beyond Image and Language.* CVPR 2026.
2. Monet official repository and customised forward.
3. Monet-7B model weights.
4. NVIDIA Isaac GR00T N1.7 official repository (action model and released configs).
5. GR00T N1.7 `Gr00tN1d7ActionHead` source.
6. GR00T N1.7 configuration.
7. LIBERO benchmark.
8. LeRobot LIBERO guide.
9. *LaST0: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model.* ICML 2026.
10. *LaRA-VLA: Latent Reasoning VLA.* ICML 2026.
11. *CoVT: Chain-of-Visual-Thought.* ECCV 2026 (Phase 2 only).
12. Hugging Face LeRobot multi-task DiT implementation (alternative training reference).
