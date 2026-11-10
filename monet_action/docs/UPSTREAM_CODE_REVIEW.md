# Review of the upstream Monet code (fork base `08939998`)

## Scope

**Read in full:**

* The parts of `monet_qwen_model/modeling_qwen2_5_vl_monet.py` that differ from stock Qwen2.5-VL:
  the `latent_mode` forward, the ce_patch pass, attention hooks, the text model and the rope index.
* The latent ("ABS-VIS") blocks of `inference/vllm/monet_gpu_model_runner.py`, plus the inference scripts.
* The latent handling in the RL actor (`RL/verl/workers/actor/dp_actor.py`) and in the RL HF model
  (`RL/monet_models/transformers/monet_modeling_qwen2_5_vl.py`).
* The latent-segment construction in `src/utils.py` and `src/main.py`.
* The trainer structure (`src/trainer.py`), the example scripts, the requirements and the README.

**Automated scans** covered `src/`, `inference/`, `monet_qwen_model/` and `RL/`, looking for debugger
leftovers, hard-coded paths and external API use.

**Not reviewed line by line:** verl framework internals beyond the latent paths, the DeepSpeed configs,
and the internals of the precompute scripts (their command-line flags were checked).

## 1. Inventory

| path | what it is | used by `monet_action`? |
|---|---|---|
| `monet_qwen_model/modeling_qwen2_5_vl_monet.py` | customised Qwen2.5-VL for transformers 4.54: per-sample `latent_mode` forward, second "ce_patch" pass, alignment losses, attention-loss hooks | **imported unchanged** under a private module name; the parity reference |
| `monet_qwen_model/apply_qwen2_5_monet.py` | replaces `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl` in `sys.modules` (global monkey patch) | no |
| `inference/` | vLLM 0.10.0 runner with latent substitution (`LATENT_START_ID/END_ID/LATENT_SIZE` env vars), loader (temperature 0.1, top_p 0.8, max_tokens 4096), example | no: vLLM is not differentiable. Its state machine is emulated for parity tests |
| `src/` | SFT stages via TRL `SFTTrainer` + DeepSpeed ZeRO-2 (see below) | no |
| `RL/` | EasyR1/verl fork for VLPO; details below | no |
| `script_examples/`, `deepspeed/` | launch scripts with placeholder paths; ZeRO-2 config | no |

The SFT stages in `src/`:

* stage 1: cross-entropy warm-up;
* stage 2: `latent_mode`, with observation-token representations aligned to precomputed teacher
  representations;
* stage 3: latent embeddings aligned to precomputed target latents, plus cross-entropy.

The VLPO pipeline in `RL/`:

* vLLM rollouts record latents through a TCP/UDP hook;
* the actor recomputes log-probs with the recorded latents patched in;
* latents get a Gaussian log-prob with σ;
* rewards come from external LLM judges (Gemini or DeepSeek).

## 2. Latent layouts: the finding that matters for Phase 1

| path | sequence around the latent segment | positions fed with the previous final (post-RMSNorm) hidden state | K used | gradient through the feedback |
|---|---|---|---|---|
| SFT data + HF `latent_mode` (`src/utils.py: replace_img_pad_with_latent_pad`; modeling file §2.2.2) | `<abs_vis_token>` + K × `<abs_vis_token_pad>` + `</abs_vis_token>` | the K pad slots (the start token keeps its embedding) | 8 in `sft_stage2/3.sh` (`--latent_size` default 4) | step input is `.detach()`ed; gradients return through the ce_patch pass |
| vLLM inference runner | sampled `<abs_vis_token>`, K-1 sampled tokens, then `</abs_vis_token>` | the **start-token slot itself** plus K-1 following slots (K total) | `LATENT_SIZE=10` (`inference/example.sh`) | none (inference) |
| RL (VLPO) rollouts + actor log-probs | as vLLM | `torch.arange(start, end)`: the start slot + K-1 inner positions. The docstring says "strictly inside", but the code includes the start | `latent_size=10` (`RL/examples/vlpo_train.sh`) | latents enter as fixed tensors (Gaussian log-prob) |

**Consequence.** The released Monet-7B (config `stage: avt_v5_stage2`) was *last* trained with the RL/vLLM
layout at K=10, which is also the layout the authors evaluate with. `monet_action` implements both layouts
differentiably:

* the default is `latent_convention: vllm`;
* `configs/d1_sft_convention.yaml` is the ablation;
* parity: against the authors' HF `latent_mode` for `sft`, and against an emulation of the runner's
  state machine for `vllm`.

Batch size: the authors' `latent_mode` loops over samples and is only tested at batch size 1. Our version
is batched with left padding.

## 3. Problems found

| severity | location | problem | action taken |
|---|---|---|---|
| high (breaks the script) | `script_examples/sft_stage3.sh:42` | `--stage "avt_v5_stage2"` is rejected by `src/utils.py:19` (`choices=['sft_stage1','sft_stage2','sft_stage3']`) | **fixed** → `sft_stage3` |
| low | `script_examples/sft_stage3.sh:51` | `.../sft_stage1${STAGE1_MODEL}` is missing `/` (stage 2 uses `sft_stage1/${...}`) | **fixed** |
| low | root `README.md` (2 links) | links to `monet_qwen_model/vllm/monet_gpu_model_runner.py`, which does not exist | **fixed** → `inference/vllm/...` |
| high (for this fork) | root `.gitignore` | ignores every directory named `scripts/` and `logs/`; the first push of `monet_action/scripts/` was silently empty | **fixed** with an override in `monet_action/.gitignore` + a hygiene test |
| junk | `transformers/.DS_Store` | macOS metadata, the only file in that directory | **removed** |
| medium (docs vs code) | `RL/verl/workers/actor/dp_actor.py` `collect_varlen_segment_indices` | the docstring says "strictly inside … skipping the first matched segment"; the code takes `arange(start, end)` for all segments | documented (section 2) |
| info | `src/utils.py:41`, scripts, inference | three different latent sizes: default 4, SFT 8, RL/inference 10 | documented; our K sweep covers 5/10/20 |
| info | `requirements.txt` vs `RL/requirements.txt` | SFT/inference: transformers 4.54.0, vllm 0.10.0. RL: transformers 4.51.3, vllm 0.8.5, torch 2.6.0, flash-attn 2.6.3. These are two incompatible environments (the authors use separate conda envs); the checkpoint's `config.json` says 4.51.3 | our venv uses torch 2.7.1 + transformers 4.54.0, the SFT/inference stack without vLLM |
| info | `src/main.py:60` | `wandb.init(entity="Latent-Think")` is hard-coded whenever `--wandb_name` is set; it fails for other W&B accounts | left as is (unused) |
| info | `RL/tools/custom_api.py`, `RL/verl/trainer/ray_trainer.py:288-291`, `RL/verl/workers/fsdp_workers.py:158-161` | external paid LLM judges (Gemini, DeepSeek) need API keys | never used; the proposal forbids paid API calls |
| info | 18 files, mostly `RL/` | 73 `pdb`/`breakpoint` references, nearly all commented out or bare imports. The only uncommented `breakpoint()` (`RL/verl/protocol.py:554`) sits inside a string literal; `src/main.py:21` imports `pdb` without using it | left as is; `monet_action` has a test that forbids debugger leftovers in its own code |
| info | `monet_qwen_model/apply_qwen2_5_monet.py` | global `sys.modules` monkey patch of transformers' Qwen2.5-VL | not used; we load the file under a private module name, so stock transformers stays intact |

## 4. What was changed in upstream files

Every change is behaviour-neutral:

* `script_examples/sft_stage3.sh`: fixed the stage name and the path.
* Root `README.md`: corrected two links and added a short fork note.
* Deleted `transformers/.DS_Store`.

The modeling file, `inference/`, `src/` and `RL/` are otherwise unchanged, so parity tests and any
future `git pull` from upstream stay meaningful.

## 5. Recommendations

* Keep upstream code verbatim. Wrap it (`monet_loader.py`) rather than editing it, and re-run
  `make test-monet` after any upstream update.
* Never import `apply_qwen2_5_monet.py` in the same process as stock Qwen2.5-VL code.
* If RL or SFT code is ever needed, build its own environment from `RL/requirements.txt` or
  `requirements.txt`; do not mix it into `~/venvs/monet_action`.
* When reporting Monet-7B behaviour, state the latent layout and K used, because the authors' own
  stages differ.
