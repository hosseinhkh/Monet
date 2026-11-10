# Developing Monet-Action

## Layout and ground rules

* `monet_action/` is our package. Everything we develop lives here.
* The upstream Monet code at the repository root (`monet_qwen_model/`, `inference/`, `src/`, `RL/`,
  `script_examples/`) is kept **verbatim** at upstream commit `08939998`.
  * `monet_qwen_model/modeling_qwen2_5_vl_monet.py` is imported unchanged and is the reference
    implementation in the parity tests. Never edit it. Wrap it instead (`monet_loader.py`,
    `monet_backbone.py`).
  * See `docs/UPSTREAM_CODE_REVIEW.md` for what the upstream code contains and its known issues.
* Results discipline:
  * Numbers in `results/` come only from runs that actually executed. The CPU evidence is labelled
    *tiny*.
  * Never commit checkpoints or data. `runs/` and `data/` are git-ignored; videos under
    `results/eval/**/videos/` too.

## Environment

```bash
bash scripts/setup_env.sh
```

```bash
source ~/venvs/monet_action/bin/activate && pip install -r environment/dev_requirements.txt && pre-commit install
```

`setup_env.sh` creates the isolated venv. The second line installs ruff and pre-commit; the hooks only
touch `monet_action/`.

## Everyday commands (`make help`)

| goal | command |
|---|---|
| fast tests (~30 s) | `make test-fast` |
| one component | `make test-tokens`, `test-monet`, `test-gradients`, `test-dit`, `test-data`, `test-tool`, `test-sim`, `test-train`, `test-dashboard` |
| real Monet-7B (GPU) | `make test-monet7b` |
| everything | `make test-all` |
| lint / autofix | `make lint` / `make format` |
| G0 audit | `make audit-cpu` (tiny) / `make audit` (Monet-7B) |
| monitor runs | `make dashboard` (then `ssh -L 8765:localhost:8765 <server>`) |
| debug zip | `make bundle RUN=runs/d1_k10` or the dashboard's download buttons |

Test markers: `slow`, `sim` (LIBERO/MuJoCo), `tokenizer` (Monet tokenizer files), `gpu` (CUDA + Monet-7B
weights). GPU and sim tests self-skip when their requirements are missing.

## Adding things

* **A new variant.** Add `configs/<name>.yaml` with `_base_: base.yaml` (or a variant) and override only
  what differs. Keep data, split, seeds, DiT and step budget matched unless the variant is *about* them.
  `test_every_config_loads_and_validates` picks the file up automatically.
* **A new intervention.**
  * Latent-only perturbations go in `conditioning.apply_latent_intervention`.
  * VL/latent masking goes in `MonetActionPolicy.encode`.
  * Add a case to `tests/test_masks.py` or `tests/test_tool_result.py`, and list it in
    `scripts/evaluate_all.sh`.
* **A new prompt token group** (Phase 2): see `docs/PHASE2_EXTENSION_POINTS.md`.
* **A new metric or log field.** Write it to the run directory following the file contract in
  `docs/DASHBOARD.md`, so the dashboard and debug bundles pick it up.

## Logging, monitoring and debugging

Every training and evaluation run writes these files (`logging_utils.RunMonitor`):

* `logs/<kind>.log`: the text log;
* `status.json`: state and step, with a heartbeat every 30 s;
* `events.jsonl`: start, resume, checkpoint, val, sim_eval, finished, failed;
* `system.jsonl`: GPU, CPU, RAM and disk usage.

Metrics go to `train_log.jsonl`, `val_log.jsonl`, `overfit_log.jsonl`, `sim_eval_log.jsonl` and, for
evaluations, `episodes.jsonl`. A crash marks `status.json` as `failed` with the traceback.

`python -m monet_action.bundle <run_dir> -o bundle.zip` packs logs, configs, metrics and environment info
for later debugging. Checkpoint weights are excluded unless you pass `--include-checkpoints`.

## Before pushing

```bash
make lint && make test-fast
```

Run `make test-all` for changes touching the model, data or the simulator.
