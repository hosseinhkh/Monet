# Environment

* `train_requirements.txt` / `sim_requirements.txt`: pinned direct dependencies (one venv holds both,
  because closed-loop evaluation runs the policy and MuJoCo in one process).
* `cpu_validation_lock.txt`: full `uv pip freeze` of the venv used for the CPU validation in this repo
  (Python 3.11.18, torch 2.7.1+cpu, no GPU).
* `server_lock.txt`: written by `scripts/setup_env.sh` on the GPU server (commit it after setup).

Torch: `torch==2.7.1` from `https://download.pytorch.org/whl/cu128` (CUDA 12.8 wheels include
Blackwell sm_100/sm_120 kernels). Rendering: `MUJOCO_GL=egl` on the NVIDIA server; `osmesa` on CPU-only
machines (needs `libosmesa6`).

The setup script creates a **new** venv (default `~/venvs/monet_action`) and never modifies existing
OpenVLA / LIBERO / MuJoCo environments.
