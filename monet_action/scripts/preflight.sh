#!/usr/bin/env bash
# One command that checks EVERYTHING before the long Phase-1 training runs, in order, stopping at the
# first failure. Each step logs to results/preflight/NN_<step>.log; a summary table is printed at the end.
#   bash scripts/preflight.sh                 # GPU server, full preflight (~1.5-2 h, dominated by the dry run)
#   bash scripts/preflight.sh --from audit    # resume from a step
#   bash scripts/preflight.sh --only render   # run one step
#   bash scripts/preflight.sh --cpu           # CPU machine: tiny model, no GPU steps (tests the preflight itself)
#   KEEP_GOING=1 bash scripts/preflight.sh    # do not stop at the first failure
source "$(dirname "$0")/common.sh"
CPU=0; FROM=""; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cpu) CPU=1 ;;
    --from) FROM="$2"; shift ;;
    --only) ONLY="$2"; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac; shift
done
if [ "$CPU" = 1 ]; then export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa DEVICE=cpu; fi
PF=results/preflight; mkdir -p "$PF"
STEPS=(env data weights render tests_cpu tests_gpu audit memory dry_run smoke)
declare -A RES
started=0; [ -z "$FROM" ] && started=1
TINY=(monet.model_path=tiny-random monet.dtype=float32 train.mixed_precision=no prompt.image_size=112)

run_step() {
  case "$1" in
    env)       python -m monet_action.preflight env $([ "$CPU" = 1 ] && echo --allow-cpu) ;;
    data)      python -m monet_action.preflight data $([ "$CPU" = 1 ] && echo --partial) ;;
    weights)   if [ "$CPU" = 1 ]; then echo "SKIP (cpu mode)"; return 3; fi
               python -m monet_action.preflight weights ;;
    render)    python -m monet_action.preflight render ;;
    tests_cpu) python -m pytest -q -rs -p no:warnings -m "not gpu" tests ;;
    tests_gpu) if [ "$CPU" = 1 ]; then echo "SKIP (cpu mode)"; return 3; fi
               python -m pytest -q -rs -p no:warnings tests/test_monet7b_gpu.py | tee "$PF/06_gpu_tests.log.raw"
               python -m monet_action.preflight gpu-tests --log "$PF/06_gpu_tests.log.raw" ;;
    audit)     if [ "$CPU" = 1 ]; then
                 python -m monet_action.audit --config configs/audit.yaml --device cpu --out "$PF/audit_tiny" --override "${TINY[@]}"
                 python -m monet_action.preflight audit --report "$PF/audit_tiny/g0_report_tiny.json"
               else
                 python -m monet_action.audit --config configs/audit.yaml --device cuda --out results/audit
                 python -m monet_action.preflight audit --report results/audit/g0_report_monet7b.json
               fi ;;
    memory)    if [ "$CPU" = 1 ]; then echo "SKIP (cpu mode)"; return 3; fi
               bash scripts/profile_memory.sh configs/d2_lora.yaml
               python -m monet_action.preflight memory ;;
    dry_run)   MODEL=$([ "$CPU" = 1 ] && echo tiny || echo real) bash scripts/dry_run.sh ;;
    smoke)     if [ "$CPU" = 1 ]; then
                 python -m monet_action.train --config configs/smoke.yaml --device cpu --override "${TINY[@]}" \
                   train.output_dir=runs/smoke_cpu train.max_steps=300 train.overfit_eval_every=50 data.num_workers=0 \
                   state.hidden_dim=128 conditioner.cond_dim=64 action_head.hidden_size=64 action_head.num_layers=2 action_head.num_heads=4
                 python -m monet_action.preflight overfit --run runs/smoke_cpu
               else
                 bash scripts/smoke_overfit.sh
                 python -m monet_action.preflight overfit --run runs/smoke
               fi ;;
  esac
}

# Step results persist in a state file so `--from` / `--only` resumes keep earlier passes.
# A full run (no --from / --only) starts from a clean state. CPU self-tests never count for the GPU.
STATE="$PF/state_$([ "$CPU" = 1 ] && echo cpu || echo gpu).tsv"
if [ -z "$FROM" ] && [ -z "$ONLY" ]; then : > "$STATE"; fi
touch "$STATE"
record() { awk -F'\t' -v k="$1" '$1 != k' "$STATE" > "$STATE.tmp"; printf "%s\t%s\n" "$1" "$2" >> "$STATE.tmp"; mv "$STATE.tmp" "$STATE"; }
i=0
for s in "${STEPS[@]}"; do
  i=$((i + 1))
  [ "$s" = "$FROM" ] && started=1
  if [ "$started" = 0 ] || { [ -n "$ONLY" ] && [ "$s" != "$ONLY" ]; }; then continue; fi
  log=$(printf "%s/%02d_%s.log" "$PF" "$i" "$s")
  echo "==> [$i/${#STEPS[@]}] $s   (log: $log)"
  t0=$(date +%s)
  set +e; ( set -e; run_step "$s" ) > "$log" 2>&1; rc=$?; set -e
  dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then RES[$s]="PASS (${dt}s)"; elif [ $rc -eq 3 ]; then RES[$s]="SKIP"; else RES[$s]="FAIL (${dt}s)"; fi
  echo "    ${RES[$s]}"
  record "$s" "${RES[$s]} $(date '+%F %T')"
  grep -E "^\s+\[(FAIL|WARN)\]" "$log" | head -8 || true
  if [ $rc -ne 0 ] && [ $rc -ne 3 ] && [ "${KEEP_GOING:-0}" != 1 ]; then
    echo; echo "STOPPED at '$s'. Last lines of $log:"; tail -25 "$log"
    echo "Fix it, then resume with: bash scripts/preflight.sh --from $s"; exit 1
  fi
done
echo; echo "================ PREFLIGHT SUMMARY ================" | tee "$PF/summary.txt"
for s in "${STEPS[@]}"; do
  if [ -n "${RES[$s]:-}" ]; then r="${RES[$s]}"
  else r="$(awk -F'\t' -v k="$s" '$1 == k {print $2 "  (earlier run)"}' "$STATE")"; r="${r:-not run}"; fi
  printf "  %-10s %s\n" "$s" "$r" | tee -a "$PF/summary.txt"
done
if grep -q "FAIL\|not run" "$PF/summary.txt"; then echo "NOT READY (see above)" | tee -a "$PF/summary.txt"; exit 1; fi
if [ "$CPU" = 1 ]; then echo "CPU PREFLIGHT OK (GPU steps skipped - run without --cpu on the server)" | tee -a "$PF/summary.txt"; exit 0; fi
if grep -q "SKIP" "$PF/summary.txt"; then echo "NOT READY: GPU steps were skipped" | tee -a "$PF/summary.txt"; exit 1; fi
echo "READY TO TRAIN. Recommended: $(cat "$PF/recommended_batch.txt" 2>/dev/null)" | tee -a "$PF/summary.txt"
echo "Next: start training in tmux (docs/RUNBOOK.md section 6): bash scripts/train_all.sh   | monitor: make dashboard"
