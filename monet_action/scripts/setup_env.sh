#!/usr/bin/env bash
# Create an ISOLATED venv for Monet-Action (does not touch existing OpenVLA/LIBERO/MuJoCo envs).
#   bash scripts/setup_env.sh                # GPU server (CUDA 12.8 wheels, Blackwell sm_120)
#   TORCH_INDEX=https://download.pytorch.org/whl/cpu bash scripts/setup_env.sh   # CPU-only
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$HOME/venvs/monet_action}"
PY="${PYTHON:-python3.11}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
THIRD_PARTY="${THIRD_PARTY:-$HOME/third_party}"
LIBERO_SHA=8f1084e3132a39270c3a13ebe37270a43ece2a01

if command -v uv >/dev/null; then uv venv -p "$PY" "$VENV"; else "$PY" -m venv "$VENV"; fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
PIP="pip"; command -v uv >/dev/null && PIP="uv pip"
$PIP install --index-url "$TORCH_INDEX" "torch==2.7.1" "torchvision==0.22.1"
$PIP install -r "$HERE/environment/train_requirements.txt"
$PIP install -r "$HERE/environment/sim_requirements.txt"

# LIBERO at a pinned commit, made importable via a .pth file (its setup.py does not
# install the nested package correctly in editable mode).
mkdir -p "$THIRD_PARTY"
if [ ! -d "$THIRD_PARTY/LIBERO" ]; then git clone https://github.com/Lifelong-Robot-Learning/LIBERO "$THIRD_PARTY/LIBERO"; fi
git -C "$THIRD_PARTY/LIBERO" checkout -q "$LIBERO_SHA"
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
echo "$THIRD_PARTY/LIBERO" > "$SITE/libero_src.pth"
# Non-interactive LIBERO config (LIBERO otherwise prompts on first import).
mkdir -p "$HOME/.libero"
if [ ! -f "$HOME/.libero/config.yaml" ]; then
  R="$THIRD_PARTY/LIBERO/libero/libero"
  printf "benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: %s/../datasets\nassets: %s/assets\n" "$R" "$R" "$R" "$R" "$R" > "$HOME/.libero/config.yaml"
fi
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
print("transformers", transformers.__version__)
PY
pip freeze > "$HERE/environment/server_lock.txt"
echo "venv ready: $VENV  (lock written to environment/server_lock.txt)"
