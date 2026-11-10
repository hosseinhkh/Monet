#!/usr/bin/env bash
# Sourced by every script. Override any variable from the environment.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
export VENV="${VENV:-$HOME/venvs/monet_action}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$HERE/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export MUJOCO_GL="${MUJOCO_GL:-egl}"          # osmesa on machines without a GPU
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
export DEVICE="${DEVICE:-cuda}"
