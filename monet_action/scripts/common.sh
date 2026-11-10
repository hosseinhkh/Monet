#!/usr/bin/env bash
# Sourced by every script. Picks the Python environment in this order:
#   1. VENV=/path/to/venv (explicit)   2. an active conda env other than "base"   3. ~/venvs/monet_action
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
if [ -n "${VENV:-}" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
elif [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-base}" != "base" ]; then
  : # use the active conda environment (e.g. `conda activate monet_action`)
elif [ -f "$HOME/venvs/monet_action/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/venvs/monet_action/bin/activate"
else
  echo "[monet_action] no environment found: run 'conda activate monet_action' (see scripts/setup_env.sh)" >&2
  exit 1
fi
export PYTHONPATH="$HERE/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export MUJOCO_GL="${MUJOCO_GL:-egl}"          # osmesa on machines without a GPU
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
export DEVICE="${DEVICE:-cuda}"
# GPU pinning: everything (PyTorch + MuJoCo EGL rendering) runs on ONE card, GPU_ID (default 0), numbered as
# in nvidia-smi. Inside Python that card is cuda:0. Set CUDA_VISIBLE_DEVICES yourself to override.
export GPU_ID="${GPU_ID:-0}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES%%,*}}"
# MONET_GPU_RESERVE=0.95 -> training/eval/audit processes claim 95% of the card right after loading Monet
# (useful on shared servers). 0 = allocate on demand (default).
export MONET_GPU_RESERVE="${MONET_GPU_RESERVE:-0}"
