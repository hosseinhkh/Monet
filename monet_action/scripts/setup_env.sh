#!/usr/bin/env bash
# Create an ISOLATED environment for Monet-Action (never modifies existing OpenVLA / LIBERO / MuJoCo envs).
#   bash scripts/setup_env.sh                                   # conda env "monet_action" if conda exists, else a venv
#   ENV_KIND=conda ENV_NAME=monet_action bash scripts/setup_env.sh
#   ENV_KIND=venv  VENV=$HOME/venvs/monet_action bash scripts/setup_env.sh
#   TORCH_INDEX=https://download.pytorch.org/whl/cpu bash scripts/setup_env.sh   # CPU-only machine
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_KIND="${ENV_KIND:-auto}"
ENV_NAME="${ENV_NAME:-monet_action}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"   # CUDA 12.8 wheels: Blackwell (sm_100/sm_120)
THIRD_PARTY="${THIRD_PARTY:-$HOME/third_party}"
LIBERO_SHA=8f1084e3132a39270c3a13ebe37270a43ece2a01
if [ "$ENV_KIND" = "auto" ]; then command -v conda >/dev/null && ENV_KIND=conda || ENV_KIND=venv; fi

if [ "$ENV_KIND" = "conda" ]; then
  eval "$(conda shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] reusing existing conda env '$ENV_NAME'"
  else
    conda create -y -n "$ENV_NAME" python=3.11
  fi
  conda activate "$ENV_NAME"
else
  VENV="${VENV:-$HOME/venvs/monet_action}"
  PY="${PYTHON:-python3.11}"
  [ -d "$VENV" ] || "$PY" -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi
set -u
python -m pip install --upgrade pip
python -m pip install --index-url "$TORCH_INDEX" "torch==2.7.1" "torchvision==0.22.1"
python -m pip install -r "$HERE/environment/train_requirements.txt" -r "$HERE/environment/sim_requirements.txt" \
                      -r "$HERE/environment/dev_requirements.txt"
# The package itself, editable, so `python -m monet_action.<tool>` works from any directory.
# (Never `pip install monet_action` from PyPI: that is not this code.)
python -m pip install -e "$HERE" --no-deps

# LIBERO at a pinned commit, importable via a .pth file (its setup.py does not install the nested package).
mkdir -p "$THIRD_PARTY"
[ -d "$THIRD_PARTY/LIBERO/.git" ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO "$THIRD_PARTY/LIBERO"
git -C "$THIRD_PARTY/LIBERO" fetch -q origin && git -C "$THIRD_PARTY/LIBERO" checkout -q "$LIBERO_SHA"
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
echo "$THIRD_PARTY/LIBERO" > "$SITE/libero_src.pth"
# Non-interactive LIBERO config (LIBERO otherwise prompts on first import).
mkdir -p "$HOME/.libero"
if [ ! -f "$HOME/.libero/config.yaml" ]; then
  R="$THIRD_PARTY/LIBERO/libero/libero"
  printf "benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: %s/../datasets\nassets: %s/assets\n" \
    "$R" "$R" "$R" "$R" "$R" > "$HOME/.libero/config.yaml"
fi
python - <<'PY'
import torch, transformers, mujoco, robosuite
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
print("transformers", transformers.__version__, "| mujoco", mujoco.__version__, "| robosuite", robosuite.__version__)
from libero.libero import benchmark  # noqa: F401
print("LIBERO import ok")
import monet_action, os
print("monet_action importable from", os.path.dirname(monet_action.__file__))
PY
python -m pip freeze > "$HERE/environment/server_lock.txt"
echo "[setup] done ($ENV_KIND). Lock written to environment/server_lock.txt."
if [ "$ENV_KIND" = "conda" ]; then echo "[setup] next: conda activate $ENV_NAME"; fi
