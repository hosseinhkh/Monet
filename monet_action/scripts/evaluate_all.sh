#!/usr/bin/env bash
# G3: full LIBERO-Spatial closed-loop evaluation (10 tasks x 3 seeds x 10 episodes), E=1 and E=2,
# D3 latent interventions + bypass diagnostics, solver sensitivity, latency, open-loop metrics, report.
# Overridable (used by scripts/dry_run.sh):
#   RUNS_DIR (runs)  EVAL_OUT (results/eval)  REPORT_OUT (results/phase1_report.md)
#   EVAL_ARGS / OFFLINE_ARGS / LATENCY_ARGS   extra CLI flags for closed_loop / offline / latency
#   BEST (d2_lora)   run used for the 2/8-step solver sensitivity
source "$(dirname "$0")/common.sh"
RUNS="${RUNS_DIR:-runs}"
OUT="${EVAL_OUT:-results/eval}"
EVAL=configs/eval.yaml
read -r -a EA <<< "${EVAL_ARGS:-}"
read -r -a OA <<< "${OFFLINE_ARGS:-}"
read -r -a LA <<< "${LATENCY_ARGS:-}"
cl() {  # cl <run> <out_name> [extra flags...]
  local run="$1" name="$2"; shift 2
  python -m monet_action.evaluate closed_loop --checkpoint "$RUNS/$run" --eval-config $EVAL \
    --output-dir "$OUT/$name" --device "$DEVICE" "${EA[@]}" "$@"
}
for run in d0_k0 d1_k10 d2_lora d0_k0_lora d4_query d4_query_lora d1_k5 d1_k20 d1_latents_only d1_no_toolresult d1_sft_convention; do
  [ -f "$RUNS/$run/last" ] || { echo "skip $run (no checkpoint in $RUNS/$run)"; continue; }
  cl "$run" "$run"
  python -m monet_action.evaluate offline --checkpoint "$RUNS/$run" --steps 2 4 8 --output-dir "$OUT/$run" --device "$DEVICE" "${OA[@]}"
  python -m monet_action.evaluate latency --checkpoint "$RUNS/$run" --steps 2 4 8 --output-dir "$OUT/$run" --device "$DEVICE" "${LA[@]}"
done
# E=2 execution prefix for the principal variants
for run in d0_k0 d1_k10 d2_lora d0_k0_lora; do
  [ -f "$RUNS/$run/last" ] && cl "$run" "${run}_E2" --execution-prefix 2
done
# D3: perturbation diagnostics on D2 (OOD by construction; not independently trained models)
if [ -f "$RUNS/d2_lora/last" ]; then
  for iv in zero mask shuffle_order truncate:5 gaussian swap mask_vl mask_all; do
    cl d2_lora "d3_${iv/:/}" --intervention "$iv"
  done
fi
# solver sensitivity (2/4/8 Euler steps) on the best checkpoint
B="${BEST:-d2_lora}"
if [ -f "$RUNS/$B/last" ]; then
  for n in 2 8; do cl "$B" "${B}_steps$n" --num-inference-steps $n; done
fi
[ -f "$RUNS/d5_mlp/last" ] && cl d5_mlp d5_mlp
python -m monet_action.report --eval-dir "$OUT" --out "${REPORT_OUT:-results/phase1_report.md}"
