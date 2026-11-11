# Runbook: from a fresh server to Phase-1 training and evaluation

Run every command from `Monet/monet_action/` with the conda env active, unless stated otherwise.
Times are estimates for one ~96 GB Blackwell GPU.

## 0. Prerequisites (once)

```bash
nvidia-smi
```
You need a driver that supports CUDA 12.8 (R570 or newer for Blackwell), and the GPU must be idle.

```bash
conda --version && git --version && df -h ~
```
You need conda, git and about 60 GB free: 7 GB data, 17 GB Monet-7B and checkpoints.

```bash
ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so* || sudo apt-get install -y libegl1 libgl1 libosmesa6
```
EGL is needed for headless MuJoCo rendering. The NVIDIA driver normally provides `libEGL_nvidia`.

## 1. Get the code

Fresh clone:
```bash
git clone https://github.com/hosseinhkh/Monet.git && cd Monet && git checkout claude/friendly-goldberg-6d4ukm
```

Existing clone:
```bash
cd Monet && git fetch origin && git checkout claude/friendly-goldberg-6d4ukm && git pull origin claude/friendly-goldberg-6d4ukm
```

```bash
cd monet_action
```

## 2. Environment (conda, isolated: existing OpenVLA/LIBERO envs are not touched)

```bash
bash scripts/setup_env.sh
```
This creates the conda env `monet_action` (Python 3.11) with torch 2.7.1 from the CUDA 12.8 wheels,
transformers 4.54.0, MuJoCo 3.3.2, robosuite 1.4.1 and LIBERO@8f1084e. It installs this package in
editable mode (`pip install -e .`) and writes `environment/server_lock.txt`. It takes about 10 minutes.

```bash
conda activate monet_action
```

If the env was created before the editable install was added to `setup_env.sh` (error
`No module named 'monet_action'`), run this once from `monet_action/`:
```bash
pip install -e . --no-deps
```
Never `pip install monet_action` from PyPI: that is not this code.

## 3. Downloads

```bash
export HF_HOME=/path/with/space/huggingface   # optional: where Monet-7B (16.6 GB) is cached
```

```bash
bash scripts/download_data.sh
```
This fetches `lerobot/libero_spatial_image` (6.5 GB, into `data/`) and Monet-7B, both at pinned revisions.

```bash
bash scripts/download_libero_raw.sh
```
Optional (0.5 GB): one raw LIBERO demo file, used by the demo-replay simulator test.

## 4. Two-minute sanity check

```bash
python -m monet_action.preflight env && python -m monet_action.preflight data && python -m monet_action.preflight weights
```
These print PASS/WARN/FAIL with numbers: GPU, bf16 TFLOP/s, package versions, 432 episodes / 52,970
frames, 4 Monet shards, tokenizer.

```bash
make test-fast
```
About 30 s; must be green.

## 5. Full preflight (one command, ~1.5–2 h)

```bash
bash scripts/preflight.sh
```

It runs these steps in order and stops at the first failure. Each step writes
`results/preflight/NN_<step>.log`, and `results/preflight/summary.txt` holds the summary.

| step | what it proves | pass criterion |
|---|---|---|
| `env` | Python/torch/transformers/MuJoCo/robosuite/LIBERO versions, GPU + bf16, disk, RAM | no FAIL |
| `data` | full LeRobot dataset present and decodable | 69 shards, 432 episodes, 52,970 frames |
| `weights` | Monet-7B snapshot complete, latent token ids, processor | 4 shards ≈ 16.6 GB, ids 151665–151667 |
| `render` | headless EGL rendering + `env.step` | non-blank images, state refresh |
| `tests_cpu` | the whole test suite except the GPU module (incl. sim, overfit, dashboard) | all pass |
| `tests_gpu` | **real Monet-7B**: no missing weights, finite `[2,10,3584]` latents, parity with the authors' code, state dependence, LoRA gradients, latency | all pass, **none skipped** |
| `audit` | G0 report: both latent layouts, parity, K=0 vs K=10, data conventions, sim orientation | `results/audit/g0_report_monet7b.json` checks pass |
| `memory` | peak memory + step time of D2 (the largest variant) at micro-batch 2/4/8/12/16, 5 steps each | recommends the largest micro-batch that fits the share **and** divides the effective batch 16 → `results/preflight/recommended_batch.txt`; a probe that crashes for a reason other than memory fails the step with its error |
| `dry_run` | every variant for 6 steps (incl. D1→D2 and D0→D0-LoRA warm starts, in-training sim eval), then the complete `evaluate_all.sh` and report | prints `DRY RUN OK` |
| `smoke` | G1: Monet-7B overfits 32 demo windows; 2-task closed-loop smoke rollout | fixed-noise loss falls, final < 0.5 × initial, gripper accuracy > 0.9 |

Useful variants:

* Resume after fixing something: `bash scripts/preflight.sh --from audit`.
* Run one step: `bash scripts/preflight.sh --only render`.
* Run all steps and list every problem: `KEEP_GOING=1 bash scripts/preflight.sh`.

When the summary prints **`READY TO TRAIN`**, look at the smoke videos in `results/smoke/closed_loop/videos/`.
The arm should move toward the bowl, and nothing should be mirrored or upside down.

The recommendation is read from the stored profile, so it can be recomputed without re-profiling (takes
seconds):
```bash
python -m monet_action.preflight memory && cat results/preflight/recommended_batch.txt
```
With a 50% share of a 96 GB card, expect `train.batch_size=8 train.grad_accum=2`. A larger micro-batch can
fit, but 16 is not a multiple of it, so it would change the effective batch the configs are tuned for.

Run long steps (the preflight, training, evaluation) inside **tmux**, so they survive SSH disconnects:
`tmux new -s <name>`, detach with `Ctrl-b d`, reattach with `tmux attach -t <name>`.

## 5b. GPU usage (what runs where)

Everything runs on **one card, GPU 0 by default**. `scripts/common.sh` sets `CUDA_VISIBLE_DEVICES=0`,
`MUJOCO_EGL_DEVICE_ID=0` (simulator rendering) and `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so "GPU 0" is the
card `nvidia-smi` calls 0. Inside Python that card is `cuda:0`. Direct `python -m monet_action.<tool>` commands
apply the same defaults. Choose another card with `GPU_ID=1`, or let the code pick the card with the most free memory
right now with `GPU_ID=auto`:
```bash
export GPU_ID=auto
```

The `env` check prints every GPU with its free memory (`*` marks the selected one) and lists other processes on the
selected card. It **fails** when the selected card has less than 40 GB free (`MONET_MIN_FREE_GB`), and then suggests a
better `GPU_ID`. Monet-7B needs 16.6 GB for its weights plus activations, and the memory step measures the real
peak. On a shared machine, check before long runs:
```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv && nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

| preflight step | device |
|---|---|
| `env` | GPU: bf16 matmul throughput, free memory, other processes on GPU 0 |
| `data`, `weights` | CPU / disk |
| `render` | GPU 0 (EGL) |
| `tests_cpu` | CPU by design (tiny-model correctness suite); its simulator tests render on GPU 0 |
| `tests_gpu`, `audit`, `memory`, `dry_run`, `smoke` | **GPU 0 with real Monet-7B** |

Training, evaluation and the in-training simulator checks all run on GPU 0.

**Sharing a card: reserve and cap a fixed share (e.g. 50% of GPU 0).** Put the settings in `env.local`. It is
git-ignored and read by every `scripts/*.sh` and every `python -m monet_action.*`; variables exported in the shell
still win.
```bash
cp env.local.example env.local
```
The example sets `GPU_ID=0`, `MONET_GPU_RESERVE=0.5` and `MONET_GPU_HOLD=1`. With that, every training, evaluation and audit process:

* **claims** 50% of GPU 0 as its first GPU action, before loading data or weights (the memory shows in `nvidia-smi`
  and is reused for the weights and activations);
* is **capped** at 50% (`torch.cuda.set_per_process_memory_fraction`). A run that would need more fails with a
  normal out-of-memory error in *your* process instead of starving the other half's users.
  `MONET_GPU_LIMIT` sets a different cap; `0` disables it.

The preflight adapts:

* `env` reports the cap, checks the reservation works, and warns if the share is larger than the memory free
  right now (others must free theirs first).
* `memory` profiles batch sizes inside the share and recommends the largest that fits within it minus 3 GB.
  `train_all.sh` then uses that batch, keeping the effective batch at 16 through gradient accumulation.

A 50% share of a 96 GB card (≈ 47 GB) holds Monet-7B (16.6 GB) plus training activations for moderate micro-batches.

**Keeping the share between runs (`MONET_GPU_HOLD=1`).** Without it, the 50% is free for anyone while nothing of
yours runs: after the preflight, between two training variants, between training and evaluation. With it, a small
background **holder** process owns the share during those gaps:

* the first script that sources `scripts/common.sh` (the preflight, `train_all.sh`, `evaluate_all.sh`, ...) starts
  the holder if it is not running. It is detached, so it survives logout and tmux, and stays until you stop it;
* each Monet-Action GPU process takes a **lease** before it touches the GPU. The holder frees its memory (≈ 1 s)
  and the process claims the 50% itself. When the process exits or crashes, the holder takes the memory back within
  a second. Others can only grab the share during those one-second hand-overs;
* the holder leaves 2 GB free and keeps a ~0.5 GB CUDA context while a run has the share. If the card is busy, it
  holds what it can (`growing`) and takes more as other jobs free memory.

```bash
bash scripts/gpu_hold.sh status    # state (holding / growing / released / stopped), held GB, live leases
```
```bash
bash scripts/gpu_hold.sh stop      # give the 50% back (e.g. when Phase 1 is finished)
```
```bash
bash scripts/gpu_hold.sh start     # start it by hand (--fraction 0.5 to override MONET_GPU_RESERVE)
```
Use a fixed `GPU_ID` with the holder: `GPU_ID=auto` would see the held memory as used and might pick another card.
The holder's state and log are in `~/.cache/monet_action/gpu_hold/gpu<N>/`.

## 6. Start training (in tmux, so it survives SSH disconnects)

**What is trained.** The action expert, conditioner and projectors are always trained in full. Monet-7B's
base weights, vision tower included, are frozen in the core runs; D2 and D0-LoRA adapt it with LoRA. The
optional capacity ladder trains more of Monet: wider LoRA, or full fine-tuning of 4 or 8 decoder layers. Why
the whole model is not fine-tuned, and when to climb the ladder: [FINETUNING.md](FINETUNING.md).

| run (config) | Monet-7B | trained | steps | starts from |
|---|---|---|---|---|
| D0 (`d0_k0`), K=0 | frozen | state + TOOL_RESULT projectors, conditioner, DiT | 20k | scratch |
| D1 (`d1_k10`), K=10 | frozen | same modules as D0 | 20k | scratch |
| D2 (`d2_lora`), K=10 | **LoRA** r=16, α=32, q/k/v/o, layers 20–27 (~2.9M params) | D1's modules + LoRA | 10k (stage B) | `runs/d1_k10/ckpt_0010000` |
| D0-LoRA (`d0_k0_lora`), K=0 | **LoRA**, same as D2 | D0's modules + LoRA | 10k (stage B) | `runs/d0_k0/ckpt_0010000` |
| D4 (`d4_query`), K=10 learned queries | frozen | D1's modules + K query embeddings | 20k | scratch |
| D5 (`d5_mlp`), K=10 | frozen | projectors, conditioner, MLP head | 20k | scratch |
| `EXTRA=1`: K=5, K=20, 3 ablations | frozen | as D1 | 20k | scratch |
| `EXTRA=1`: D4-LoRA (`d4_query_lora`) | **LoRA** | D4's modules + LoRA | 10k (stage B) | `runs/d4_query/ckpt_0010000` |
| capacity ladder `lora_wide` (`d2_lora_wide` + `d0_k0_lora_wide`) | **LoRA** r=64, attention + MLP, all 28 layers (~161M) | D2's / D0-LoRA's modules + wide LoRA | 10k (stage B) each | D1@10k / D0@10k |
| capacity ladder `ft4` (`d2_ft4` + `d0_k0_ft4`) | **full fine-tuning** of layers 24–27 (~0.93B, fp32, lr 1e-5) | same modules + those layers | 10k (stage B) each | D1@10k / D0@10k |
| capacity ladder `ft8` (`d2_ft8` + `d0_k0_ft8`) | **full fine-tuning** of layers 20–27 (~1.86B); whole card | same modules + those layers | 10k (stage B) each | D1@10k / D0@10k |

Every run uses an effective batch of 16 windows and the same data split, seed, DiT and conditioner. The
step-10k checkpoints of D0, D1 and D4 are never deleted (`train.keep_every=10000`), because the LoRA runs
start from them.

```bash
tmux new -s phase1
```

```bash
conda activate monet_action && cd /path/to/Monet/monet_action && mkdir -p runs && bash scripts/train_all.sh 2>&1 | tee -a runs/train_all.log
```

* **Batch size.** It comes automatically from `results/preflight/recommended_batch.txt`, the same for
  every variant. Override it with `TRAIN_OVERRIDES="train.batch_size=8 train.grad_accum=2"`.
* **Order.** Variants run in dependency order: D0 → D1 → D2 (from D1@10k) → D0-LoRA (from D0@10k) → D4
  → D5. Add `EXTRA=1` for the K sweep {5,20}, the ablations and D4-LoRA.
* **tmux.** Detach with `Ctrl-b d` and reattach with `tmux attach -t phase1`.
* **One variant at a time.** Use `bash scripts/train_d1.sh` (likewise `train_d0`, `d0_lora`, `d2`, `d4`,
  `d5`), or `python -m monet_action.train --config configs/<run>.yaml`. These use the configs' batch
  (8 × 2) unless you add `--override train.batch_size=... train.grad_accum=...`.
* **How long.** Each run's ETA is on the dashboard and in `runs/<run>/status.json`. Before starting, a
  rough per-run estimate is steps × grad_accum × s/step (from `results/preflight/memory_profile.json`
  at your micro-batch), plus about 10% for validation and simulator checks.
* **GPU share (`MONET_GPU_HOLD=1`).** Each run takes the 50% over from the holder when it starts and
  hands it back when it ends. Between variants the holder keeps it, so other users cannot take it.
* **Capacity ladder (after the core runs).** Run it with `bash scripts/train_capacity.sh lora_wide`
  (then `ft4`; `ft8` needs the whole card), or with `CAPACITY="lora_wide ft4" bash scripts/train_all.sh`.
  * Each level profiles its memory once, then trains the K=10 run and its K=0 control at the same
    micro-batch.
  * Full fine-tuning checkpoints are large: ~11 GB each for `ft4`, ~22 GB for `ft8`. Check disk space.

Monitoring (second tmux window, `Ctrl-b c`):
```bash
conda activate monet_action && cd /path/to/Monet/monet_action && make dashboard
```

From your laptop:
```bash
ssh -N -L 8765:localhost:8765 <user>@<gpu-server>
```
Then open http://localhost:8765.

**Healthy run**

* `status.json` is running and its heartbeat is under 1 min old.
* Training loss drops quickly in the first 1–2k steps.
* `grad_norm` is finite. The first step's gradient audit passed for every trained group:
  `action_head`, `conditioner`, `state_projector`, `monet_lora` (D2).
* `val_log`: `action_mse` decreases and `gripper_acc` rises above 0.9.
* `sim_eval_log` (every 5k steps, development init states): success rises above 0.
* GPU utilisation stays high.

**Red flags → action**

| red flag | action |
|---|---|
| state `stale` | the process died: `tail runs/<run>/logs/train.log`, then re-run the same command (it resumes) |
| `failed` | traceback in the dashboard / `status.json`; make a debug zip with `make bundle RUN=runs/<run>` |
| val loss rising while train loss falls (overfitting) | keep the best checkpoint (`ckpt_*` at the val minimum); consider `--override data.augment=true` |
| `sim_eval_failed` events | rendering problem (training continues); check `MUJOCO_GL=egl`, `MUJOCO_EGL_DEVICE_ID` |
| out of memory | lower `train.batch_size` and raise `train.grad_accum`, or `monet.recurrence_impl=recompute monet.gradient_checkpointing=true` |

Interruptions:

* Re-run `bash scripts/train_all.sh`. Finished runs return immediately, and the interrupted run resumes
  from `runs/<run>/last` with the same model, optimiser, RNG and data position.
* `Ctrl-C` marks the run `stopped`.

## 7. Evaluate and report

```bash
bash scripts/evaluate_all.sh 2>&1 | tee -a results/evaluate_all.log
```
This runs about 25 closed-loop evaluations: 10 tasks × 3 seeds × 10 episodes on init states 0–29, E=1
and E=2, the D3 interventions and 2/8 solver steps. It also runs offline and latency profiles, and
finally writes `results/phase1_report.md`. Evaluation resumes from `episodes.jsonl` when re-run.

```bash
python -m monet_action.evaluate compare results/eval/d0_k0 results/eval/d1_k10 results/eval/d0_k0_lora results/eval/d2_lora
```
Paired statistics: bootstrap CI and McNemar p-value.

Then fill in `RESEARCH_MEMO.md` from the report, keeping negative results.

When Phase 1 is finished, give the reserved GPU share back to the other users of the card:
```bash
bash scripts/gpu_hold.sh stop
```

## 8. Debugging and sharing logs

```bash
make bundle RUN=runs/d2_lora
```
Produces a zip of logs, configs, metrics, environment and pip freeze, without weights. The dashboard has
the same buttons (with or without checkpoints). Single checks can be re-run at any time, e.g.
`python -m monet_action.preflight render` or `make test-sim`.

## 9. Troubleshooting

| symptom | fix |
|---|---|
| `EGL` / `eglQueryString` / `PLATFORM.EGL is None` | `export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`; install `libegl1`; with several GPUs also set `MUJOCO_EGL_DEVICE_ID=0` |
| LIBERO asks "Do you want to specify a custom path…" | `~/.libero/config.yaml` is missing; re-run `scripts/setup_env.sh` |
| `Weights only load failed` | use the repository code (`LiberoEnv` loads init states with `weights_only=False`) |
| `torch.cuda.is_available() == False` | the driver is too old for CUDA 12.8, or torch is a CPU wheel; check `python -c "import torch; print(torch.version.cuda)"` |
| HF download stalls or 401 | `huggingface-cli login`, or set `HF_TOKEN`; then retry `scripts/download_data.sh` (it resumes) |
| `tests_gpu` skipped | CUDA is not visible or the weights are not in the HF cache; set `MONET7B_PATH=/path/to/Monet-7B` |
| `TypeError` / crash in the `memory` step, all micro-batches failing | the step now names the crash; it is a bug, not memory. Send `results/preflight/08_memory.log` (or a bundle) |
| GPU share not handed over / run starts with less memory than expected | `bash scripts/gpu_hold.sh status`; a process that never exits keeps its lease (`leases`). See `~/.cache/monet_action/gpu_hold/gpu0/holder.log` |
