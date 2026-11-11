#!/usr/bin/env bash
# Monet adaptation capacity ladder (docs/FINETUNING.md): a K=10 run and its matched K=0 control at one level
# of Monet adaptation, stage B from the frozen stage-A checkpoints (D1@10k / D0@10k, as D2 / D0-LoRA).
#   bash scripts/train_capacity.sh lora_wide   # LoRA r=64 on attention + MLP of all 28 layers (fits a 50% share)
#   bash scripts/train_capacity.sh ft4         # full fine-tuning of decoder layers 24-27 (~+15 GB over D2)
#   bash scripts/train_capacity.sh ft8         # full fine-tuning of layers 20-27 (~+30 GB: whole card, MONET_GPU_RESERVE=0.9)
# The micro-batch is profiled once per level (results/preflight/memory_profile_d2_<level>.json) and kept at the
# effective batch 16. Re-running resumes like every training run.
source "$(dirname "$0")/common.sh"
LEVEL="${1:?usage: bash scripts/train_capacity.sh lora_wide|ft4|ft8}"
K10="d2_$LEVEL"; K0="d0_k0_$LEVEL"
[ -f "configs/$K10.yaml" ] && [ -f "configs/$K0.yaml" ] || { echo "unknown level '$LEVEL' (configs/$K10.yaml)"; exit 2; }
for ck in runs/d1_k10/ckpt_0010000 runs/d0_k0/ckpt_0010000; do
  [ -d "$ck" ] || { echo "missing stage-A checkpoint $ck: train D1 and D0 first (scripts/train_all.sh)"; exit 1; }
done
PF=results/preflight
REC="$PF/recommended_batch_$K10.txt"
if [ ! -f "$REC" ]; then
  echo "==== $(date '+%F %T') profiling memory for $K10 (once)"
  PROFILE_OUT="$PF/memory_profile_$K10.json" bash scripts/profile_memory.sh "configs/$K10.yaml"
  python -m monet_action.preflight memory --summary "$PF/memory_profile_$K10.json"
fi
read -r -a OV <<< "$(cat "$REC")"
for c in "$K10" "$K0"; do  # same micro-batch for the pair (the K=0 control needs less memory)
  echo "==== $(date '+%F %T') training $c ${OV[*]}"
  python -m monet_action.train --config "configs/$c.yaml" --device "$DEVICE" --override "${OV[@]}"
done
echo "==== $(date '+%F %T') capacity level '$LEVEL' done. Evaluate: bash scripts/evaluate_all.sh (includes these runs)"
