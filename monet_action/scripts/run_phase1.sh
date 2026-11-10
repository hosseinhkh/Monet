#!/usr/bin/env bash
# Whole Phase-1 pipeline on one GPU. Each stage is resumable; rerun after interruptions.
source "$(dirname "$0")/common.sh"
bash scripts/audit_upstream.sh
bash scripts/smoke_overfit.sh
bash scripts/train_d0.sh
bash scripts/train_d1.sh
bash scripts/train_d2.sh      # stage B: warm-starts from runs/d1_k10/ckpt_0010000 (stage A)
bash scripts/train_d0_lora.sh # matched control for D2 (K=0 + same LoRA); needs runs/d0_k0/ckpt_0010000
bash scripts/train_d4.sh
bash scripts/train_d5.sh
if [ "${EXTRA:-0}" = "1" ]; then   # K sweep {5,20} and ablations (latents-only, no TOOL_RESULT action)
  for c in d1_k5 d1_k20 d1_latents_only d1_no_toolresult d1_sft_convention d4_query_lora; do
    python -m monet_action.train --config configs/$c.yaml --device "$DEVICE"
  done
fi
bash scripts/evaluate_all.sh
