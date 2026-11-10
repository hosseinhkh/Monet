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
transformers 4.54.0, MuJoCo 3.3.2, robosuite 1.4.1 and LIBERO@8f1084e. It also writes
`environment/server_lock.txt`. It takes about 10 minutes.

```bash
conda activate monet_action
```

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
| `memory` | peak memory + step time of D2 at micro-batch 4/8/12/16 | recommends a batch → `results/preflight/recommended_batch.txt` |
| `dry_run` | every variant for 6 steps (incl. D1→D2 and D0→D0-LoRA warm starts, in-training sim eval), then the complete `evaluate_all.sh` and report | prints `DRY RUN OK` |
| `smoke` | G1: Monet-7B overfits 32 demo windows; 2-task closed-loop smoke rollout | fixed-noise loss falls, final < 0.5 × initial, gripper accuracy > 0.9 |

Useful variants:

* Resume after fixing something: `bash scripts/preflight.sh --from audit`.
* Run one step: `bash scripts/preflight.sh --only render`.
* Run all steps and list every problem: `KEEP_GOING=1 bash scripts/preflight.sh`.

When the summary prints **`READY TO TRAIN`**, look at the smoke videos in `results/smoke/closed_loop/videos/`.
The arm should move toward the bowl, and nothing should be mirrored or upside down.

## 5b. GPU usage (what runs where)

Everything runs on **one card, GPU 0 by default**. `scripts/common.sh` sets `CUDA_VISIBLE_DEVICES=0`,
`MUJOCO_EGL_DEVICE_ID=0` (simulator rendering) and `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so "GPU 0" is the
card `nvidia-smi` calls 0. Inside Python that card is `cuda:0`. Use another card with `GPU_ID=1 bash scripts/...`.

| preflight step | device |
|---|---|
| `env` | GPU: bf16 matmul throughput, free memory, other processes on GPU 0 |
| `data`, `weights` | CPU / disk |
| `render` | GPU 0 (EGL) |
| `tests_cpu` | CPU by design (tiny-model correctness suite); its simulator tests render on GPU 0 |
| `tests_gpu`, `audit`, `memory`, `dry_run`, `smoke` | **GPU 0 with real Monet-7B** |

Training, evaluation and the in-training simulator checks all run on GPU 0.

**Claiming the whole card.** PyTorch normally allocates memory as needed. On a shared server, reserve
it up front instead:
```bash
export MONET_GPU_RESERVE=0.95
```
Every training, evaluation and audit process then holds 95% of GPU 0 right after loading Monet. It shows
in `nvidia-smi` as used by that process and is reused for activations; 3 GB stays free for CUDA kernels
and EGL rendering. Measured memory peaks (`memory` step) are unaffected. The `env` step verifies that the
reservation works.

## 6. Start training (in tmux, so it survives SSH disconnects)

```bash
tmux new -s phase1
```

```bash
conda activate monet_action && cd ~/Monet/monet_action && mkdir -p runs && bash scripts/train_all.sh 2>&1 | tee -a runs/train_all.log
```
The batch size comes automatically from `results/preflight/recommended_batch.txt`. Variants run in
dependency order: D0 → D1 → D2 (from D1@10k) → D0-LoRA (from D0@10k) → D4 → D5. Detach with `Ctrl-b d`
and reattach with `tmux attach -t phase1`. Add `EXTRA=1` for the K sweep {5,20}, the ablations and
D4-LoRA.

Monitoring (second tmux window, `Ctrl-b c`):
```bash
conda activate monet_action && cd ~/Monet/monet_action && make dashboard
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
