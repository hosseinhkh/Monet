#!/usr/bin/env bash
# All Phase-1 training runs on one GPU, in dependency order (D2 needs D1@10k, D0-LoRA needs D0@10k).
# Every run resumes from its last checkpoint when this script is re-run after an interruption.
#   bash scripts/train_all.sh                       # core: D0, D1, D2, D0-LoRA, D4, D5
#   EXTRA=1 bash scripts/train_all.sh               # + K sweep {5,20}, ablations, D4-LoRA
#   CAPACITY="lora_wide ft4" bash scripts/train_all.sh   # + Monet capacity ladder (wider LoRA, full fine-tuning)
#   TRAIN_OVERRIDES="train.batch_size=16 train.grad_accum=1" bash scripts/train_all.sh   # e.g. from preflight
source "$(dirname "$0")/common.sh"
read -r -a OV <<< "${TRAIN_OVERRIDES:-$(cat results/preflight/recommended_batch.txt 2>/dev/null || true)}"
run() {
  echo "==== $(date '+%F %T') training $1 ${OV[*]:-}"
  python -m monet_action.train --config "configs/$1.yaml" --device "$DEVICE" ${OV[@]:+--override "${OV[@]}"}
}
run d0_k0
run d1_k10
run d2_lora        # stage B from runs/d1_k10/ckpt_0010000
run d0_k0_lora     # stage B from runs/d0_k0/ckpt_0010000 (matched control for D2)
run d4_query
run d5_mlp
if [ "${EXTRA:-0}" = "1" ]; then
  for c in d1_k5 d1_k20 d1_latents_only d1_no_toolresult d1_sft_convention d4_query_lora; do run "$c"; done
fi
# Monet adaptation capacity ladder (docs/FINETUNING.md), each level = K=10 run + matched K=0 control:
#   CAPACITY="lora_wide ft4" bash scripts/train_all.sh     (ft8 needs the whole card: MONET_GPU_RESERVE=0.9)
for level in ${CAPACITY:-}; do bash scripts/train_capacity.sh "$level"; done
echo "==== $(date '+%F %T') all training done. Next: bash scripts/evaluate_all.sh"
