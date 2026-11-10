# Monet-Action v2 — Phase 1

Monet-7B's self-fed continuous latent reasoning used as the context of a callable, compact GR00T-style
flow-matching DiT action tool, trained by behaviour cloning and evaluated closed-loop on LIBERO-Spatial.

```
RGB (agent + wrist) + instruction + 8D state
  -> RobotStateProjector: 2 state tokens placed inside the prompt (state visible BEFORE reasoning)
  -> Monet customised Qwen2.5-VL prefill (authors' modeling file, unchanged)
  -> K genuine self-fed latent steps: input[slot k] = final hidden state of slot k-1     [B,K,3584]
  -> LatentToActionConditioner: NULL + VL tokens + ordered latents, typed, masked       [B,N,512]
  -> CALL_ACTION (typed tool) -> FlowMatchingDiT (+ state encoder), 4 Euler steps       [B,8,7]
  -> execute E=1|2 actions with env.step -> fresh observation = TOOL_RESULT -> repeat
```

**Status (honest summary).** The code, configs, tests and scripts for G0–G3 are complete. Everything that
can run without a GPU was run in this repository's CPU container:

* the test suite (51 tests);
* the G0 audit (real data, real LIBERO simulator, and Monet's real modeling code with a *tiny random*
  configuration);
* a tiny overfit;
* a short real-LIBERO closed-loop trace.

**Not yet run (needs the 96 GB GPU):**

* any Monet-7B forward pass;
* D0–D5 training;
* the closed-loop LIBERO-Spatial evaluation;
* latency and cost numbers.

There are **no task-success results** in this repository yet. Nothing here should be read as evidence for
or against the hypothesis.

## Quick start on the GPU server

```bash
cd Monet/monet_action
bash scripts/setup_env.sh          # isolated venv ~/venvs/monet_action (torch 2.7.1 cu128, transformers 4.54.0, LIBERO@8f1084e)
bash scripts/download_data.sh      # lerobot/libero_spatial_image@d86c0b9 (~6.5 GB) + Monet-7B@39c1e32 (~16.6 GB)
bash scripts/audit_upstream.sh     # G0 on the REAL Monet-7B -> results/audit/g0_report_monet7b.{json,md} + fast tests
bash scripts/profile_memory.sh configs/d2_lora.yaml   # pick micro-batch from measured peak memory
bash scripts/smoke_overfit.sh      # G1: overfit 32 windows with Monet-7B + 2-task closed-loop smoke rollout
bash scripts/train_d0.sh           # D0  frozen Monet, K=0
bash scripts/train_d1.sh           # D1  frozen Monet, K=10
bash scripts/train_d2.sh           # D2  LoRA (layers 20-27, q/k/v/o, r=16), K=10: 10k joint steps from D1@10k (budget-matched)
bash scripts/train_d4.sh           # D4  matched-compute learned-query control (K=10 slots, no self-feedback)
bash scripts/train_d5.sh           # D5  MLP head engineering control
bash scripts/evaluate_all.sh       # G3: 10 tasks x 3 seeds x 10 eps, E=1 & E=2, D3 interventions, 2/4/8 steps, latency, report
# or everything: bash scripts/run_phase1.sh   (every stage is resumable)
```

All training commands resume deterministically when re-run. Resume restores the model, optimiser, RNG and
data-sampler position from `runs/<name>/last`. Evaluation also resumes, skipping episodes already in
`episodes.jsonl`. Any config key can be overridden, e.g.
`python -m monet_action.train --config configs/d1_k10.yaml --override train.batch_size=12 monet.num_latents=5`.

CPU-only validation (what was run here): `bash scripts/cpu_validation.sh`.

## Variants (identical data, split, DiT, conditioner, seeds, step budget)

| id | Monet | K | reasoning | trainable | config |
|---|---|---|---|---|---|
| D0 | frozen | 0 | none | state projector + conditioner + DiT | `configs/d0_k0.yaml` |
| D1 | frozen | 10 | genuine self-fed recurrence | same as D0 (identical parameter count, tested) | `configs/d1_k10.yaml` |
| D2 | LoRA q/k/v/o, layers 20–27 | 10 | self-fed, FM loss back-propagated through all K steps | D1 + LoRA. Stage A = D1's first 10k steps; stage B = 10k joint steps (20k head updates total, same as D0/D1) | `configs/d2_lora.yaml` |
| D3 | D2 at eval | 10 | latents zeroed / masked / order-shuffled / truncated / Gaussian / swapped across tasks | — (diagnostic, OOD) | `--intervention ...` |
| D4 | frozen | 10 slots | trainable query embeddings at the same slots, no feedback | D1 + K×3584 queries | `configs/d4_query.yaml` |
| D5 | frozen | 10 | self-fed | MLP regression head (control only) | `configs/d5_mlp.yaml` |
| ablation | frozen | 10 | self-fed | conditioning sees latents only | `configs/d1_latents_only.yaml` |

## Repository map

```
monet_action/
  README.md  AUDIT.md (G0 report)  SOURCE_VERSIONS.md  RESEARCH_MEMO.md (pre-registered analysis, to fill)
  environment/  train/sim requirements, cpu_validation_lock.txt (server_lock.txt written on setup)
  configs/      base + d0_k0 d1_k10 d2_lora d4_query d5_mlp d1_latents_only, audit, smoke, eval, eval_smoke, tiny_cpu
  src/monet_action/
    monet_loader.py     loads the authors' modeling file under a private name; tiny random Monet for tests
    prompt.py           chat-format prompt, image tokens, state placeholders, latent slots (sft|vllm conventions)
    monet_backbone.py   MonetLatentReasoner (cached / recompute recurrence, bptt|truncated, learned_query D4)
                        + reference emulations of the authors' latent_mode forward and vLLM runner (parity only)
    state_context.py    RobotStateProjector (8D -> state tokens in Monet embedding space)
    conditioning.py     LatentToActionConditioner, feature_source switch, D3 interventions, LatentBank
    dit_action.py       FlowMatchingDiT (GR00T N1.7 contract, compact), DiT/MLP action heads
    flow_matching.py    time sampling, interpolation, velocity target, masked loss, Euler integrator
    normalization.py    q01/q99 action + z-score state normalisation, gripper sign handling
    libero_dataset.py   LeRobot parquet frames, episode-disjoint split, H=8 windows with masks, collator
    libero_env.py       LIBERO wrapper -> PolicyObservation (images rot180, 8D state, instruction only)
    tool_runtime.py     typed CALL_ACTION / TOOL_RESULT, lock-step batched closed-loop runner
    policy.py           MonetActionPolicy (assembly, loss, predict, LoRA injection, checkpoints)
    train.py            training loop, gradient audit, val/overfit curves, deterministic resume
    evaluate.py         closed_loop | offline | latency | compare
    metrics.py          offline metrics, Wilson CI, paired bootstrap, McNemar, cost aggregation
    report.py           assembles results/phase1_report.md from evaluation outputs
    audit.py            G0 audit (sources, Monet latents + parity, data batch, simulator I/O)
  tests/              51 tests (see below)
  scripts/            setup, download, audit, smoke, train_d{0,1,2,4,5}, evaluate_all, run_phase1, cpu_validation
  results/            audit/ (G0 reports), cpu_validation/ (tiny-model evidence)
```

The original Monet code outside `monet_action/` is left untouched. It is the reference implementation that
the parity tests compare against.

## Critical tests → files

| # | requirement | tests |
|---|---|---|
| 1 | genuine self-fed recurrence vs soft prompt / hidden-state extraction | `test_monet_latents.py::test_self_feed_property`, `test_parity_with_authors_latent_mode`, `test_parity_with_vllm_runner_state_machine`, `test_not_a_soft_prompt_or_plain_hidden_extraction`, `test_conventions_differ` |
| 2 | train/inference parity, fixed seeds, masks | `test_cached_equals_recompute_values_and_grads`, `test_batch_vs_single_sample_parity`, `test_train_mode_equals_eval_mode`, `test_train_smoke.py` (reload reproduces metrics) |
| 3 | state dependence of reasoning states | `test_state_dependence_of_reasoning` |
| 4 | FM loss → DiT → conditioner → LoRA | `test_gradients.py` (+ runtime gradient audit asserted on the first training step of every run) |
| 5 | interpolation, velocity target, timestep encoding | `test_flow_matching.py` |
| 6 | 2/4/8-step integration, deterministic noise | `test_euler_exact_on_conditional_flow`, `test_euler_deterministic_noise_and_step_counts` |
| 7 | normalisation / gripper | `test_normalization.py`, `test_data.py::test_gripper_and_action_conventions` |
| 8 | padding masks, horizon truncation, no cross-episode leakage | `test_masks.py` |
| 9 | CALL_ACTION, env stepping, feedback record, observation refresh | `test_tool_loop.py` (synthetic env + **real LIBERO** `test_real_libero_call_action_cycle`) |
| 10 | latent dropout/shuffle sensitivity, no oracle data | `test_latent_shuffle_changes_prediction`, `test_interventions`, `test_policy_observation_has_no_oracle_fields` |

Run them with `pytest -q tests`. Marks: `-m "not slow"`, `-m sim`. Tests need the Monet tokenizer files
(`MONET_TOKENIZER_PATH` or HF access) and the dataset (`MONET_ACTION_DATA`, default `data/libero_spatial_image`).

## Key design decisions (all configurable)

* **Default latent convention: `sft`.** This is the authors' differentiable HF path. The vLLM-runner
  convention is available as `monet.latent_convention=vllm`; AUDIT.md §1 explains why they differ.
* **`feedback_grad=bptt`.** No detach anywhere in the loop. `truncated` reproduces the authors' SFT
  gradient.
* **Recurrence implementations.** The KV-cache recurrence is the default. `recompute` is identical up to
  float error and is the only one compatible with HF gradient checkpointing.
* **Conditioning.**
  * Tokens: all valid prompt tokens (final layer), the K latents, and a NULL token.
  * Feature sources: `latents_only`, `vl_only`, `vl_plus_latents`.
  * Matched variants: D0 and D1 share every parameter and differ only in K.
* **Flow matching.** GR00T schedule: t ~ (1-Beta(1.5,1))·0.999, 1000 buckets, 4 Euler steps (2/4/8
  evaluated).
* **Normalisation.** Actions use q01/q99 → [-1, 1] on the train split only. The gripper is kept at ±1 and
  sign-thresholded before `env.step`.
* **Evaluation protocol.**
  * Benchmark init states are matched across variants: seed *i* uses states [10i, 10i+10).
  * The seed also sets the DiT noise.
  * 10 parallel environments run in lock-step with batched CALL_ACTION. Per-sample latency is also
    profiled at batch size 1 (`evaluate latency`).

## Expected resource use (estimates, not measurements)

* **Training throughput.** A training sample runs a ~220-token Monet-7B prefill plus 10 single-token
  steps, a backward pass to the state projector and LoRA, and the vision tower without grad. That is
  roughly 10 TFLOP per sample, so 20k steps × 16 samples is a few GPU-hours per variant on one Blackwell.
* **Memory.** bf16 Monet weights use about 16.6 GB. Activations depend on batch size: measure with
  `scripts/profile_memory.sh` before long runs rather than trusting this estimate. If memory is tight,
  switch to `monet.recurrence_impl=recompute` with `monet.gradient_checkpointing=true`.
* **Evaluation size.** The full protocol is 300 episodes × up to 220 steps per variant at E=1, i.e. up to
  about 66k policy calls (batched ×10).

## Known limitations / open risks

* Monet-7B has not been executed by this code yet. GPU G0 is the first thing to run, and its report must
  show finite latents, no missing weights, and parity errors at bf16 noise level.
* The real vLLM runner is not used. Parity is checked against an exact emulation of its state machine.
* No real-time or physical-safety claims. Results are paused-simulator task success.
* CoVT, SAM/DINO/depth teachers, routers, planners, RL and ASPIRE are out of scope (Phase 2+). The
  conditioner's typed-token interface is where perceptual thought tokens would be added in Phase 2.
