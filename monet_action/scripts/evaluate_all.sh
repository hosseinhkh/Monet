#!/usr/bin/env bash
# G3: full LIBERO-Spatial closed-loop evaluation (10 tasks x 3 seeds x 10 episodes), D3 interventions,
# solver sensitivity, latency profiles, open-loop metrics and the comparison report.
source "$(dirname "$0")/common.sh"
EVAL=configs/eval.yaml
OUT=results/eval
for run in d0_k0 d1_k10 d2_lora d4_query d1_k5 d1_k20 d1_latents_only d1_no_toolresult; do
  [ -d runs/$run ] || { echo "skip $run (no checkpoint)"; continue; }
  python -m monet_action.evaluate closed_loop --checkpoint runs/$run --eval-config $EVAL --output-dir $OUT/$run --device "$DEVICE"
  python -m monet_action.evaluate offline --checkpoint runs/$run --steps 2 4 8 --output-dir $OUT/$run --device "$DEVICE"
  python -m monet_action.evaluate latency --checkpoint runs/$run --steps 2 4 8 --output-dir $OUT/$run --device "$DEVICE"
done
# E=2 execution prefix for the principal variants
for run in d0_k0 d1_k10 d2_lora; do
  [ -d runs/$run ] && python -m monet_action.evaluate closed_loop --checkpoint runs/$run --eval-config $EVAL \
    --execution-prefix 2 --output-dir $OUT/${run}_E2 --device "$DEVICE"
done
# D3: perturbation diagnostics on D2 (OOD by construction; not independently trained models)
if [ -d runs/d2_lora ]; then
  # latent interventions + bypass diagnostics (mask_vl: no VL context; mask_all: unconditioned head)
  for iv in zero mask shuffle_order truncate:5 gaussian swap mask_vl mask_all; do
    python -m monet_action.evaluate closed_loop --checkpoint runs/d2_lora --eval-config $EVAL \
      --intervention "$iv" --output-dir "$OUT/d3_${iv/:/}" --device "$DEVICE"
  done
  # solver sensitivity (2/4/8 Euler steps) on the best checkpoint (default D2; override BEST=...)
  for n in 2 8; do
    python -m monet_action.evaluate closed_loop --checkpoint "runs/${BEST:-d2_lora}" --eval-config $EVAL \
      --num-inference-steps $n --output-dir "$OUT/${BEST:-d2_lora}_steps$n" --device "$DEVICE"
  done
fi
[ -d runs/d5_mlp ] && python -m monet_action.evaluate closed_loop --checkpoint runs/d5_mlp --eval-config $EVAL --output-dir $OUT/d5_mlp --device "$DEVICE"
python -m monet_action.report --eval-dir $OUT --out results/phase1_report.md
