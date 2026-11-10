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
# Machine-local settings (git-ignored; see env.local.example). Variables already set in the shell win.
if [ -f "$HERE/env.local" ]; then
  while IFS='=' read -r _k _v; do
    _k="${_k#export }"; _k="${_k// /}"
    [[ "$_k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue   # skips comments, blank and malformed lines
    _v="${_v%%#*}"; _v="${_v%"${_v##*[![:space:]]}"}"; _v="${_v#"${_v%%[![:space:]]*}"}"; _v="${_v#\"}"; _v="${_v%\"}"
    if [ -z "${!_k+x}" ]; then export "$_k=$_v"; fi
  done < "$HERE/env.local"
  unset _k _v
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
if [ "$GPU_ID" = "auto" ]; then  # the card with the most free memory right now
  GPU_ID=$( (nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "0, 0") \
            | sort -t, -k2 -nr | head -n 1 | cut -d, -f1 | tr -d ' ')
  echo "[monet_action] GPU_ID=auto -> GPU $GPU_ID" >&2
fi
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES%%,*}}"
# MONET_GPU_RESERVE=0.5 -> training/eval/audit processes claim 50% of the card as their first GPU action
# (useful on shared servers). 0 = allocate on demand (default).
export MONET_GPU_RESERVE="${MONET_GPU_RESERVE:-0}"
# MONET_GPU_HOLD=1 -> keep that share claimed between runs too: a background holder (scripts/gpu_hold.sh)
# owns it while no Monet-Action process runs and hands it over to each one (see src/monet_action/gpu_hold.py).
export MONET_GPU_HOLD="${MONET_GPU_HOLD:-0}"
if [ "$MONET_GPU_HOLD" = 1 ] && [ "$MONET_GPU_RESERVE" != 0 ] && [ "$DEVICE" != cpu ] && [ "${MONET_GPU_HOLD_AUTOSTART:-1}" = 1 ]; then
  python -m monet_action.gpu_hold ensure || echo "[monet_action] GPU holder not started (see bash scripts/gpu_hold.sh status)" >&2
fi
