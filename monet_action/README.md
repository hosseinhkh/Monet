# Monet-Action v2 — Phase 1

Monet-7B's self-fed continuous latent reasoning is used as the context of a callable, compact GR00T-style
flow-matching DiT action tool. The model is trained by behaviour cloning and evaluated closed-loop on
LIBERO-Spatial.

```
RGB (agent + wrist) + instruction + 8D state + TOOL_RESULT (previously executed action)
  -> state tokens + TOOL_RESULT tokens placed inside the prompt (visible BEFORE reasoning)
  -> Monet customised Qwen2.5-VL prefill (authors' modeling file, unchanged)
  -> K genuine self-fed latent steps: input[slot k] = final hidden state of slot k-1     [B,K,3584]
  -> LatentToActionConditioner: NULL + VL tokens + ordered latents, typed, masked       [B,N,512]
  -> CALL_ACTION (typed tool) -> FlowMatchingDiT (+ state encoder), 4 Euler steps       [B,8,7]
  -> execute E=1|2 actions with env.step -> TOOL_RESULT (fresh observation + executed actions) -> repeat
```

## The research in one page

This repository implements Phase 1 of the research proposal *Monet-Action v2: continuous latent reasoning
with a callable flow-matching DiT action expert*. The full plan, and where each part of it lives in the code,
is in **[docs/PROPOSAL.md](docs/PROPOSAL.md)**.

* **Question.** Do Monet's self-fed continuous latent reasoning states improve closed-loop robotic
  manipulation when they condition a learned continuous-action expert, compared with a matched model
  without those reasoning steps? **And** does any gain pay for its extra GPU time and latency?
* **Why it is new (a hypothesis, not a result).** Monet-7B reasons in K continuous steps, each fed the
  previous step's final hidden state. Phase 1 uses those internal states as the context of an *executable
  action tool* (CALL_ACTION → flow-matching DiT → actions → TOOL_RESULT), not only as features. Coupling a
  VLM to a DiT alone would not be new. Related latent-VLA work: LaST0 and LaRA-VLA.
* **Hypotheses.**
  * **H1 feasibility:** the latents form a trainable, useful conditioning interface.
  * **H2 added value:** K=10 self-fed steps beat matched K=0 controls (D1 vs D0, D2 vs D0-LoRA) and the
    compute-matched D4.
  * **H3 dependence:** perturbing the latents (D3) changes action quality.
  * **H4 cost:** the gain is worth the latency and GPU-seconds per success.
* **Boundaries.**
  * A compact GR00T-*style* DiT trained from scratch; not the full GR00T model or its weights.
  * CALL_ACTION is fixed: no learned tool router (Phase 3).
  * No CoVT perceptual thoughts (Phase 2).
  * Paused-simulator success only: no real-time or safety claims.
  * Positive, null and negative results all count, and all are reported.
* **Plan.**
  * Gate 0: sources, environment, latent parity, a correct LIBERO batch.
  * Gate 1: unit tests, overfit, one closed-loop step.
  * Gate 2: D0–D5 trained and evaluated on 10 tasks × 3 seeds × 10 episodes, with D3 interventions and
    latency / cost profiles.
  * Then the research conclusion in [RESEARCH_MEMO.md](RESEARCH_MEMO.md), whose analysis plan was fixed
    before any result.

## Status

The code, configs, tests, monitoring and scripts for G0–G3 are complete.

**On the GPU server (one 96 GB Blackwell card, 50% of GPU 0 reserved):** the full preflight
(`bash scripts/preflight.sh`) passed every step and printed `READY TO TRAIN`. That includes:

* the real Monet-7B tests (complete weights, finite latents, parity with the authors' code, LoRA gradients);
* the G0 audit;
* the memory profile;
* the real-model dry run of every variant and of the whole evaluation pipeline;
* the G1 smoke overfit.

Its logs and reports stay on the server, in `results/preflight/` and `results/audit/`.

**In this repository's CPU container:**

* the full test suite: 192 passed, 1 skipped (the Monet-7B GPU module, which needs CUDA). See `make test-all` and `results/cpu_validation/pytest_all.txt`;
* the whole pipeline dry run (`MODEL=tiny bash scripts/dry_run.sh`): 6 training variants, warm-start chains, in-training simulator checks, all 21 evaluation runs and the report;
* the G0 audit (`results/audit/g0_report_tiny.*`);
* a tiny overfit;
* a real-LIBERO closed-loop trace;
* a replay of recorded LIBERO demonstrations that solves the task in our simulator build.

The "tiny" runs use Monet's *real* modeling code with a tiny random configuration.

**Not yet run:** the D0–D5 training runs, the closed-loop LIBERO-Spatial evaluation, and the latency and
cost measurements. There are **no task-success results** yet. Nothing here is evidence for or against the
hypothesis.

## Quick start on the GPU server

The full step-by-step guide, with pass criteria, monitoring and troubleshooting, is
**[docs/RUNBOOK.md](docs/RUNBOOK.md)**. The short version:

```bash
git clone https://github.com/hosseinhkh/Monet.git && cd Monet && git checkout claude/friendly-goldberg-6d4ukm && cd monet_action
bash scripts/setup_env.sh && conda activate monet_action   # isolated conda env (torch 2.7.1 cu128, transformers 4.54.0, LIBERO@8f1084e)
bash scripts/download_data.sh      # lerobot/libero_spatial_image@d86c0b9 (~6.5 GB) + Monet-7B@39c1e32 (~16.6 GB)
cp env.local.example env.local     # optional, shared card: GPU_ID=0, reserve + cap 50% (MONET_GPU_RESERVE), keep it between runs (MONET_GPU_HOLD)
tmux new -s phase1                 # long steps below: run them inside tmux (detach Ctrl-b d, reattach: tmux attach -t phase1)
bash scripts/preflight.sh          # env, data, weights, render, all tests, REAL Monet-7B tests, audit, memory, dry run, smoke -> READY TO TRAIN
bash scripts/train_all.sh          # D0, D1, D2 (from D1@10k), D0-LoRA (from D0@10k), D4, D5; EXTRA=1 adds K sweep + ablations
bash scripts/evaluate_all.sh       # 10 tasks x 3 seeds x 10 eps, E=1/2, D3 + bypass interventions, 2/4/8 steps, latency, report
make dashboard                     # monitoring on 127.0.0.1:8765 (from your laptop: ssh -N -L 8765:localhost:8765 <server>)
bash scripts/gpu_hold.sh stop      # when Phase 1 is done: give the reserved GPU share back
```

* **Sharing the GPU.** With `env.local` from the example, every run claims 50% of GPU 0 before it loads
  anything and is capped there. A background holder keeps that share between runs until
  `bash scripts/gpu_hold.sh stop`. See RUNBOOK §5b.

* **Resuming.** Re-running a training command resumes deterministically: model, optimiser, RNG and
  data-sampler position come back from `runs/<name>/last`. Evaluation resumes from `episodes.jsonl`.
* **Overrides.** Any config key can be overridden:
  `python -m monet_action.train --config configs/d1_k10.yaml --override train.batch_size=12 monet.num_latents=5`.
* **Developer workflow.** Per-component tests, lint and pre-commit are described in
  [DEVELOPMENT.md](DEVELOPMENT.md).

**Start here:**

* [docs/PROPOSAL.md](docs/PROPOSAL.md): the research plan, hypotheses, work packages and gates, with a
  traceability table from every proposal requirement to its implementation and status.
* [docs/BIG_PICTURE.md](docs/BIG_PICTURE.md): the diagrams, which comparison answers which question, the
  pass criteria before long GPU runs, the proposal assessment, the power analysis, the compute budget, and
  how to read the results.
* [docs/FINETUNING.md](docs/FINETUNING.md): how much of Monet to train (frozen, LoRA, full fine-tuning),
  and why.

## Monitoring, logs and debug bundles

Every training and evaluation run writes:

* `logs/<kind>.log`: the text log, including warnings and tracebacks;
* `status.json`: state, step, ETA and errors, refreshed by a 30 s heartbeat;
* `events.jsonl`: start, resume, checkpoint, val, sim_eval, finished, failed;
* `system.jsonl`: GPU utilisation and memory, CPU, RAM and disk;
* metric streams: `train_log.jsonl` (loss, grad norms per group, LR, step and data time, memory),
  `val_log.jsonl`, `overfit_log.jsonl` and `sim_eval_log.jsonl`;
* for evaluations: `episodes.jsonl`, `traces/`, `videos/` and `summary.json`.

A crash marks the run `failed` in `status.json`, with the traceback. TensorBoard is optional
(`train.tensorboard=true`).

**Dashboard.** Run `make dashboard` (or `bash scripts/dashboard.sh`). It is a dependency-free local web
UI that works offline:

* run list with live state and progress;
* loss, validation, LR, gradient, timing, memory and GPU charts;
* log tail and search, events and files;
* evaluation tables with per-task results, failure tags and videos;
* a run comparison view and the audit reports.

**Debug bundles.** The dashboard's download buttons, or
`python -m monet_action.bundle <run_or_eval_dir> -o debug.zip`, produce a zip of logs, configs, metrics
and environment info for later debugging. Checkpoint weights are added only with
`--include-checkpoints`. Details: [docs/DASHBOARD.md](docs/DASHBOARD.md).

## Variants (identical data, split, DiT, conditioner, seeds, step budget)

| id | Monet | K | reasoning | trainable | config |
|---|---|---|---|---|---|
| D0 | frozen | 0 | none | state + TOOL_RESULT projectors, conditioner, DiT | `d0_k0.yaml` |
| D1 | frozen | 10 | genuine self-fed recurrence | same as D0 (identical parameter count, tested) | `d1_k10.yaml` |
| D2 | LoRA q/k/v/o, layers 20–27 | 10 | self-fed; FM loss back-propagated through all K steps | D1 + LoRA. Stage A = D1's first 10k steps; stage B = 10k joint steps (20k head updates total) | `d2_lora.yaml` |
| D0-LoRA | LoRA q/k/v/o, layers 20–27 | 0 | none | D0 + LoRA, same stage-A/B structure as D2 | `d0_k0_lora.yaml` |
| D4-LoRA (optional) | LoRA | 10 slots | learned queries, no feedback | D4 + LoRA | `d4_query_lora.yaml` |
| D3 | D2 at eval | 10 | latents zeroed / masked / order-shuffled / truncated / Gaussian / swapped; `mask_vl` and `mask_all` bypass checks | — (diagnostic, OOD) | `--intervention …` |
| D4 | frozen | 10 slots | trainable query embeddings at the same slots, no feedback | D1 + K×3584 queries | `d4_query.yaml` |
| D5 | frozen | 10 | self-fed | MLP regression head (control only) | `d5_mlp.yaml` |
| K sweep | frozen | 5, 20 | self-fed | as D1 | `d1_k5.yaml`, `d1_k20.yaml` |
| ablations | frozen | 10 | self-fed | latents-only conditioning; no previous action in TOOL_RESULT; SFT-stage latent layout | `d1_latents_only.yaml`, `d1_no_toolresult.yaml`, `d1_sft_convention.yaml` |
| capacity ladder (exploratory) | wide LoRA (r=64, attention + MLP, all layers) / full fine-tuning of layers 24–27 or 20–27 | 10, each with a K=0 control | self-fed | D2's modules with more of Monet trained; stage B from D1@10k / D0@10k | `d2_lora_wide`, `d2_ft4`, `d2_ft8` (+ `d0_k0_*`), see [docs/FINETUNING.md](docs/FINETUNING.md) |

**What is trained.** The action expert (DiT), conditioner and projectors are always trained in full, from
scratch. Monet-7B's base weights, vision tower included, stay frozen in every core variant; D2 adapts Monet
through LoRA:

* **Frozen variants (D0, D1, D4, D5, K sweep, ablations).** Only the small task modules train: the state
  and TOOL_RESULT projectors, the conditioner, and the action head (DiT; an MLP for D5). D4 also trains its
  K learned query embeddings. Each runs 20k steps.
* **LoRA variants (D2, D0-LoRA, D4-LoRA).** Same modules plus LoRA adapters: r=16, α=32, dropout 0.05, on
  q/k/v/o of decoder layers 20–27, the last 8 of 28. That is about 2.9M adapter parameters.
  * Stage A is the first 10k steps of the frozen run (D1, or D0 for D0-LoRA).
  * Stage B is 10k joint steps from that checkpoint, so every variant gets 20k head updates.
* **Batch.** Every variant uses an effective batch of 16 windows (micro-batch × gradient accumulation;
  the preflight picks the micro-batch). Learning rates: 1e-4 for the head and projectors, 5e-5 for LoRA.
* **More of Monet (capacity ladder).** If D2's small LoRA may be the limit, `scripts/train_capacity.sh`
  trains wider LoRA (~161M params) or fully fine-tunes 4 or 8 decoder layers (~0.93B / ~1.86B params, fp32,
  lr 1e-5). Each level gets a matched K=0 control, so the latent steps stay the only difference.
  * Fully fine-tuning all of Monet is not offered: with AdamW it needs ~104 GB beyond the weights, more
    than one 96 GB GPU.
  * It would also risk overwriting the pretrained reasoning under test.
  * Details, memory arithmetic and the pre-registered decision rule: [docs/FINETUNING.md](docs/FINETUNING.md).

An optional external baseline (official GR00T N1.7 LIBERO checkpoint, separate environment, not
comparable as-is) is described in [docs/EXTERNAL_BASELINES.md](docs/EXTERNAL_BASELINES.md).

## Repository map

```
monet_action/
  README.md  DEVELOPMENT.md  AUDIT.md (G0 report)  SOURCE_VERSIONS.md  RESEARCH_MEMO.md (pre-registered plan)
  Makefile    one target per component test, audits, dashboard, bundle, lint
  docs/       PROPOSAL.md (research plan + traceability), BIG_PICTURE.md, FINETUNING.md (how much of Monet to train),
              RUNBOOK.md, DASHBOARD.md, UPSTREAM_CODE_REVIEW.md, EXTERNAL_BASELINES.md, PHASE2_EXTENSION_POINTS.md
  environment/ train/sim/dev requirements, cpu_validation_lock.txt (server_lock.txt written on setup)
  configs/    base + d0_k0 d0_k0_lora d1_k5 d1_k10 d1_k20 d2_lora d4_query d4_query_lora d5_mlp d1_latents_only d1_no_toolresult d1_sft_convention,
              capacity ladder d2_lora_wide d2_ft4 d2_ft8 (+ matched d0_k0_lora_wide d0_k0_ft4 d0_k0_ft8),
              audit, smoke, eval, eval_smoke, tiny_cpu
  src/monet_action/
    monet_loader.py     loads the authors' modeling file under a private name; tiny random Monet for tests
    prompt.py           chat prompt, image tokens, state + TOOL_RESULT placeholders, latent slots, sanitisation
    monet_backbone.py   MonetLatentReasoner (cached / recompute, bptt|truncated, learned_query D4, split timings)
                        + reference emulations of the authors' latent_mode forward and vLLM runner (parity only)
    state_context.py    RobotStateProjector (also used for the TOOL_RESULT action tokens)
    conditioning.py     LatentToActionConditioner, feature_source switch, D3 interventions, LatentBank
    dit_action.py       FlowMatchingDiT (GR00T N1.7 contract, compact), DiT/MLP action heads
    flow_matching.py    time sampling, interpolation, velocity target, masked loss, Euler integrator
    normalization.py    q01/q99 action + z-score state normalisation, gripper sign handling
    libero_dataset.py   LeRobot parquet frames, episode-disjoint split, H=8 windows + previous actions, collator
    libero_env.py       LIBERO wrapper -> PolicyObservation; reproducible (seeded) resets
    tool_runtime.py     typed CALL_ACTION / TOOL_RESULT, lock-step batched rollouts, failure tags
    policy.py           MonetActionPolicy (assembly, loss, predict, LoRA, retention regulariser, checkpoints)
    train.py            training loop, gradient audit, val/overfit/sim-eval, representation report, resume
    evaluate.py         closed_loop | offline | latency | compare
    metrics.py          offline metrics, Wilson CI, paired bootstrap, McNemar, cost + latency aggregation
    report.py           results/phase1_report.md from evaluation outputs
    audit.py            G0 audit (sources, Monet latents + parity, data batch, simulator I/O)
    logging_utils.py    RunMonitor: logs, status heartbeat, events, system metrics
    preflight.py        PASS/WARN/FAIL checks behind scripts/preflight.sh (env, data, weights, render, tests, audit, memory, overfit)
    runtime_env.py      env.local + GPU pinning (GPU_ID -> CUDA_VISIBLE_DEVICES, MUJOCO_EGL_DEVICE_ID) before torch starts
    gpu.py              setup_gpu: holder hand-over, per-process cap (MONET_GPU_LIMIT), up-front claim (MONET_GPU_RESERVE)
    gpu_hold.py         background holder keeping the GPU share claimed between runs (scripts/gpu_hold.sh)
    dashboard/          local monitoring web app (stdlib server + vanilla JS)
    bundle.py           debug zip bundles (CLI + dashboard)
  tests/              see the table below; markers slow / sim / tokenizer / gpu
  scripts/            setup_env, download_data, preflight, profile_memory, dry_run, smoke_overfit, train_all, train_capacity, train_d{0,0_lora,1,2,4,5},
                      evaluate_all, run_phase1, gpu_hold, dashboard, audit_upstream, cpu_validation (all source common.sh)
  env.local.example   machine-local settings (copy to env.local, git-ignored): GPU_ID, MONET_GPU_RESERVE, MONET_GPU_HOLD, HF_HOME
  results/            audit/ (G0 reports), cpu_validation/ (tiny-model evidence)
```

The upstream Monet code outside `monet_action/` is left untouched. It is the reference implementation the
parity tests compare against.

## Tests → requirements

| # | requirement | tests (`make <target>`) |
|---|---|---|
| 1 | genuine self-fed recurrence vs soft prompt / hidden-state extraction | `test_monet_latents.py` (`test-monet`); real Monet-7B: `test_monet7b_gpu.py` (`test-monet7b`) |
| 2 | train/inference parity, fixed seeds, masks | `test_cached_equals_recompute_values_and_grads`, `test_batch_vs_single_sample_parity`, `test_train_mode_equals_eval_mode`, `test_train_smoke.py` |
| 3 | state dependence of reasoning states | `test_state_dependence_of_reasoning` (+ GPU version) |
| 4 | FM loss → DiT → conditioner → LoRA | `test_gradients.py` (`test-gradients`), `test_lora_gradients_on_real_monet` (GPU), runtime gradient audit |
| 5–6 | flow matching, time encodings, 2/4/8-step integration, deterministic noise | `test_flow_matching.py` (`test-dit`) |
| 7 | normalisation / gripper | `test_normalization.py`, `test_data.py` (`test-data`) |
| 8 | padding masks, horizon truncation, no cross-episode leakage | `test_masks.py`, `test_dataset_previous_actions_respect_episode_boundaries` |
| 9 | CALL_ACTION, env stepping, TOOL_RESULT feedback, observation refresh | `test_tool_loop.py`, `test_tool_result.py` (`test-tool`) |
| 10 | latent dropout/shuffle sensitivity, bypass checks, no oracle data | `test_latent_shuffle_changes_prediction`, `test_bypass_interventions_change_prediction`, `test_policy_observation_has_no_oracle_fields` |
| — | Monet-7B tokenizer, special tokens, prompt layout, injection safety | `test_tokens.py` (`test-tokens`) |
| — | LIBERO/robosuite: controller, deterministic resets, orientation, demo replay | `test_libero_sim.py` (`test-sim`) |
| — | logging contract, dashboard, bundles, repository hygiene | `test_logging.py`, `test_dashboard.py`, `test_repo_hygiene.py` |
| — | capacity ladder: wide LoRA on attention + MLP, full fine-tuning of decoder layers (fp32, bf16 autocast), matched configs, stage-B warm start, report | `test_finetune_capacity.py`, `test_full_finetune_stage_b_from_frozen_stage_a` |

## Key design decisions (all configurable)

* **Latent layout.** The default is `vllm`, the layout the released Monet-7B was last trained with
  (VLPO RL, K=10) and the one the authors use for inference. The alternative `sft` is the SFT-stage layout
  (K=8 in the authors' scripts) and is available as an ablation (`d1_sft_convention.yaml`). Both are
  differentiable here; see AUDIT.md §1.
* **Gradients.** `feedback_grad=bptt` puts no detach anywhere in the loop. `truncated` reproduces the
  authors' SFT gradient.
* **Recurrence implementation.** The KV-cache recurrence is the default. `recompute` is equivalent and is
  compatible with gradient checkpointing.
* **TOOL_RESULT.**
  * The next prompt carries the fresh images and state plus the last executed action. This is a token
    projected before reasoning, as in proposal §2.
  * During training, the previous action is hidden for 30% of samples (`tool_result.dropout`), which
    mitigates the behaviour-cloning "copycat" shortcut.
  * `d1_no_toolresult.yaml` measures whether action history helps.
* **Conditioning.**
  * Tokens: all prompt tokens (final layer), the K latents, and a NULL token.
  * Feature sources: `latents_only`, `vl_only`, `vl_plus_latents`.
  * Matched variants: D0 and D1 share every parameter.
* **Flow matching.** GR00T schedule: t ~ (1-Beta(1.5,1))·0.999, 1000 buckets, 4 Euler steps (2/4/8
  evaluated).
* **Evaluation protocol.**
  * Initial states: final reporting uses benchmark init states 0..29 (seed *i* → [10i, 10i+10)).
    Development rollouts use 30..49.
  * Resets are seeded per (task, init state), so every variant sees pixel-identical starting
    observations.
  * The seed also sets the DiT noise.
  * Execution: 10 environments run in lock-step with batched CALL_ACTION, reused across seeds.
  * Frames: simulator frames get a JPEG round-trip (`eval.jpeg_quality=90`), matching the JPEG history
    of the training images.
  * Optional closed-loop checks during training (`train.sim_eval_every`, on development states, never
    fatal).

## Expected resource use (estimates, not measurements)

* **Training throughput.** About 10 TFLOP per sample: a ~230-token Monet-7B prefill, 10 latent steps and
  a backward pass to the projectors and LoRA. 20k steps × 16 samples is a few GPU-hours per variant.
  The measured s/step per micro-batch is in `results/preflight/memory_profile.json`, and each run's ETA is
  on the dashboard.
* **Memory.** bf16 weights use about 16.6 GB. Measure activations with `scripts/profile_memory.sh`. If
  memory is tight, switch to `monet.recurrence_impl=recompute` with
  `monet.gradient_checkpointing=true`.
* **Evaluation size.** The full protocol is up to about 66k policy calls per variant at E=1, batched ×10.

## Known limitations / open risks

* Monet-7B runs correctly in this code on the GPU server (preflight `tests_gpu` and `audit`), but no
  training run is complete yet. If a run misbehaves, re-run `bash scripts/preflight.sh --only tests_gpu`
  first. It must show complete weights, finite latents and bf16-level parity.
* HF `generate()` does not work with the authors' model class. Their `forward` computes logits only for
  their SFT loss, and their inference uses vLLM. This code never needs text generation; the sanity check
  decodes greedily through the text decoder (`monet_loader.greedy_text_decode`).
* The real vLLM runner is not used. Parity is checked against an exact emulation of its state machine.
* Action history in TOOL_RESULT can cause copycat behaviour. It is mitigated with dropout and measured
  by the ablation.
* No real-time or physical-safety claims. Results are paused-simulator task success.
* CoVT, perception teachers, routers, planners, RL and ASPIRE are out of scope (Phase 2+). Extension
  points are listed in [docs/PHASE2_EXTENSION_POINTS.md](docs/PHASE2_EXTENSION_POINTS.md).
