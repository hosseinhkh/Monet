#!/usr/bin/env bash
# Measure peak memory / step time for candidate micro-batch sizes before long runs (96 GB budget).
source "$(dirname "$0")/common.sh"
CFG="${1:-configs/d2_lora.yaml}"
for B in 4 8 12 16; do
  python -m monet_action.train --config "$CFG" --device "$DEVICE" --override \
    train.output_dir=runs/memprobe_b$B train.max_steps=5 train.batch_size=$B train.grad_accum=1 \
    train.log_every=1 train.val_every=1000000 train.save_every=1000000 train.init_from=null data.num_workers=4 \
    || echo "B=$B failed (likely OOM)"
  cat runs/memprobe_b$B/memory_profile.json 2>/dev/null || true
  rm -rf runs/memprobe_b$B
done
