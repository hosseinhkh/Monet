#!/usr/bin/env bash
# Everything that can be validated without a GPU: tests, G0 audit with the tiny Monet,
# tiny overfit and a short real-LIBERO closed-loop trace (OSMesa rendering).
export MUJOCO_GL=osmesa DEVICE=cpu
source "$(dirname "$0")/common.sh"
pytest -q tests | tee results/cpu_validation/pytest_all.txt
python -m monet_action.audit --config configs/audit.yaml --device cpu --out results/audit \
  --override monet.model_path=tiny-random monet.dtype=float32 prompt.image_size=112
python -m monet_action.train --config configs/tiny_cpu.yaml --device cpu \
  --override train.output_dir=runs/overfit train.max_steps=400 train.overfit_eval_every=50 data.max_windows=32
python -m monet_action.evaluate closed_loop --checkpoint runs/overfit --device cpu --seeds 0 --task-ids 6 \
  --episodes-per-task 2 --num-parallel-envs 2 --max-steps 24 --execution-prefix 2 --save-videos 1 --init-state-offset 30 \
  --output-dir results/cpu_validation/closed_loop_tiny --tag tiny_cpu_G1
