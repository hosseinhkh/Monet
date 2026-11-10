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
