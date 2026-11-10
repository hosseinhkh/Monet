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

## Status

The code, configs, tests, monitoring and scripts for G0–G3 are complete. Everything that can run without a
GPU was run in this repository's CPU container:

* the full test suite: 170 passed, 1 skipped (the Monet-7B GPU module, which needs CUDA). See `make test-all` and `results/cpu_validation/pytest_all.txt`;
* the whole pipeline dry run (`MODEL=tiny bash scripts/dry_run.sh`): 6 training variants, warm-start chains, in-training simulator checks, all 21 evaluation runs and the report;
* the G0 audit (`results/audit/g0_report_tiny.*`);
* a tiny overfit;
* a real-LIBERO closed-loop trace;
* a replay of recorded LIBERO demonstrations that solves the task in our simulator build.

The "tiny" runs use Monet's *real* modeling code with a tiny random configuration.

**Not yet run (needs the 96 GB GPU):**

* any Monet-7B forward pass (`make test-monet7b`, `make audit`);
* D0–D5 training;
* the closed-loop LIBERO-Spatial evaluation;
* latency and cost numbers.

There are **no task-success results** yet. Nothing here is evidence for or against the hypothesis.

## Quick start on the GPU server

The full step-by-step guide, with pass criteria, monitoring and troubleshooting, is
**[docs/RUNBOOK.md](docs/RUNBOOK.md)**. The short version:

```bash
git clone https://github.com/hosseinhkh/Monet.git && cd Monet && git checkout claude/friendly-goldberg-6d4ukm && cd monet_action
bash scripts/setup_env.sh && conda activate monet_action   # isolated conda env (torch 2.7.1 cu128, transformers 4.54.0, LIBERO@8f1084e)
bash scripts/download_data.sh      # lerobot/libero_spatial_image@d86c0b9 (~6.5 GB) + Monet-7B@39c1e32 (~16.6 GB)
bash scripts/preflight.sh          # env, data, weights, render, all tests, REAL Monet-7B tests, audit, memory, dry run, smoke -> READY TO TRAIN
bash scripts/train_all.sh          # D0, D1, D2 (from D1@10k), D0-LoRA (from D0@10k), D4, D5; EXTRA=1 adds K sweep + ablations
bash scripts/evaluate_all.sh       # 10 tasks x 3 seeds x 10 eps, E=1/2, D3 + bypass interventions, 2/4/8 steps, latency, report
make dashboard                     # monitoring on 127.0.0.1:8765 (from your laptop: ssh -N -L 8765:localhost:8765 <server>)
```

* **Resuming.** Re-running a training command resumes deterministically: model, optimiser, RNG and
  data-sampler position come back from `runs/<name>/last`. Evaluation resumes from `episodes.jsonl`.
* **Overrides.** Any config key can be overridden:
  `python -m monet_action.train --config configs/d1_k10.yaml --override train.batch_size=12 monet.num_latents=5`.
* **Developer workflow.** Per-component tests, lint and pre-commit are described in
  [DEVELOPMENT.md](DEVELOPMENT.md).

**Start here:** [docs/BIG_PICTURE.md](docs/BIG_PICTURE.md). It covers the diagrams, which comparison
answers which question, the pass criteria before long GPU runs, the proposal assessment, the power
analysis, the compute budget, and how to read the results.

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

An optional external baseline (official GR00T N1.7 LIBERO checkpoint, separate environment, not
comparable as-is) is described in [docs/EXTERNAL_BASELINES.md](docs/EXTERNAL_BASELINES.md).

## Repository map

```
monet_action/
  README.md  DEVELOPMENT.md  AUDIT.md (G0 report)  SOURCE_VERSIONS.md  RESEARCH_MEMO.md (pre-registered plan)
  Makefile    one target per component test, audits, dashboard, bundle, lint
  docs/       DASHBOARD.md, UPSTREAM_CODE_REVIEW.md, EXTERNAL_BASELINES.md, PHASE2_EXTENSION_POINTS.md
  environment/ train/sim/dev requirements, cpu_validation_lock.txt (server_lock.txt written on setup)
  configs/    base + d0_k0 d0_k0_lora d1_k5 d1_k10 d1_k20 d2_lora d4_query d4_query_lora d5_mlp d1_latents_only d1_no_toolresult d1_sft_convention,
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
    dashboard/          local monitoring web app (stdlib server + vanilla JS)
    bundle.py           debug zip bundles (CLI + dashboard)
  tests/              see the table below; markers slow / sim / tokenizer / gpu
  scripts/            setup, downloads, audit, smoke, train_d{0,1,2,4,5}, evaluate_all, run_phase1, dashboard, cpu_validation
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
* **Memory.** bf16 weights use about 16.6 GB. Measure activations with `scripts/profile_memory.sh`. If
  memory is tight, switch to `monet.recurrence_impl=recompute` with
  `monet.gradient_checkpointing=true`.
* **Evaluation size.** The full protocol is up to about 66k policy calls per variant at E=1, batched ×10.

## Known limitations / open risks

* Monet-7B has not been executed by this code yet. Run `make test-monet7b` and `make audit` first; they
  must show complete weights, finite latents and bf16-level parity.
* The real vLLM runner is not used. Parity is checked against an exact emulation of its state machine.
* Action history in TOOL_RESULT can cause copycat behaviour. It is mitigated with dropout and measured
  by the ablation.
* No real-time or physical-safety claims. Results are paused-simulator task success.
* CoVT, perception teachers, routers, planners, RL and ASPIRE are out of scope (Phase 2+). Extension
  points are listed in [docs/PHASE2_EXTENSION_POINTS.md](docs/PHASE2_EXTENSION_POINTS.md).
