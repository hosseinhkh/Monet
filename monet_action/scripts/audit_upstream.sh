#!/usr/bin/env bash
# G0 on the GPU server: record environment + upstream SHAs, verify REAL Monet-7B latents
# (K=0 vs K=10, parity with the authors' latent_mode forward and the vLLM-runner state machine,
# state dependence, LoRA gradients, latency), one LIBERO batch and simulator I/O.
source "$(dirname "$0")/common.sh"
mkdir -p results/audit
for repo in https://github.com/NOVAglow646/Monet https://github.com/NVIDIA/Isaac-GR00T https://github.com/Lifelong-Robot-Learning/LIBERO; do
  echo "$repo $(git ls-remote "$repo" HEAD | cut -f1)"
done | tee results/audit/upstream_heads_now.txt
python -m monet_action.audit --config configs/audit.yaml --out results/audit --device "$DEVICE" "$@"
pytest -q -rs -p no:warnings tests/test_monet7b_gpu.py | tee results/audit/pytest_monet7b_gpu.txt
pytest -q -rs -p no:warnings -m "not gpu" tests | tee results/audit/pytest_all_but_gpu.txt
