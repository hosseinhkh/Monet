# Source and dependency versions (Phase 1)

All upstream sources were inspected at the commits below on 2026-11-09/10. `python -m monet_action.audit`
re-records the live environment into `results/audit/g0_report_*.json`; `scripts/audit_upstream.sh` also
records the current upstream HEADs so drift can be detected.

| Source | Pinned revision | How it is used | License |
|---|---|---|---|
| Monet code, github.com/NOVAglow646/Monet (this repo's upstream) | `08939998d3d643a73a316e349faa34f420429153` | `monet_qwen_model/modeling_qwen2_5_vl_monet.py` is imported **unchanged** under a private module name; the vLLM runner (`inference/vllm/monet_gpu_model_runner.py`) is the reference for the inference convention | see upstream |
| Monet-7B checkpoint, hf.co/NOVAglow646/Monet-7B | `39c1e320a3e15c357bff911fda263923edbdda37` | backbone weights + tokenizer/processor (`config.json` declares transformers 4.51.3; loads with 4.54.0 through the class' `_checkpoint_conversion_mapping`) | MIT (model card) |
| NVIDIA Isaac-GR00T N1.7, github.com/NVIDIA/Isaac-GR00T | `7aa0bca50f1097f8b17f5c274c98264b72a1419f` | **design reference only**: `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`, `gr00t/model/modules/dit.py`, `gr00t/model/modules/embodiment_conditioned_mlp.py`, `gr00t/configs/model/gr00t_n1d7.py` were read and re-implemented; no code imported, no weights loaded | Apache-2.0 |
| LIBERO, github.com/Lifelong-Robot-Learning/LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | simulator tasks, BDDL files, fixed init states (installed via `.pth`, see `scripts/setup_env.sh`) | MIT |
| Dataset, hf.co/datasets/lerobot/libero_spatial_image | `d86c0b94922572b3b657e1d1a3d01f0952ddeb46` | 432 demonstrations / 52,970 frames / 10 tasks, LeRobot v3 parquet, 256x256 agent + wrist RGB, 8D state, 7D action | Apache-2.0 |
| LeRobot LIBERO guide, huggingface.co/docs/lerobot/libero | (docs) | state / image conventions cross-checked against the data and simulator (see AUDIT.md) | - |

## Validated software (CPU validation container, no GPU)

Python 3.11.18; torch 2.7.1+cpu; torchvision 0.22.1+cpu; transformers 4.54.0; tokenizers 0.21.4;
peft 0.21.2; accelerate 1.15.0; safetensors 0.8.0; huggingface-hub 0.36.2; numpy 2.2.6; pyarrow 26.0.1;
mujoco 3.3.2; robosuite 1.4.1; bddl 1.0.1; gym 0.25.2; PyOpenGL 3.1.10 (OSMesa rendering).
Full lock: `environment/cpu_validation_lock.txt`.

## GPU server (to be recorded by `scripts/setup_env.sh` + `scripts/audit_upstream.sh`)

Target: 1x NVIDIA Blackwell, ~96 GB. torch 2.7.1 from the cu128 index (CUDA 12.8 runtime; Blackwell kernels
included). NVIDIA driver, CUDA runtime, GPU name and compute capability are written to
`results/audit/g0_report_monet7b.json` (`environment` section) and `environment/server_lock.txt`.
Not yet recorded: this repository has not been run on the target GPU.
