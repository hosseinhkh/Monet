#!/usr/bin/env bash
# End-to-end PIPELINE dry run: every training variant for a few steps, then the complete evaluate_all.sh
# with 1 task x 1 seed x 1 short episode, then the report. Catches integration bugs before days of GPU time.
#   MODEL=real bash scripts/dry_run.sh     # on the GPU server with Monet-7B (~1 h, mostly model loads)
#   MODEL=tiny bash scripts/dry_run.sh     # anywhere (CPU, tiny random Monet; ~15 min)
# Nothing here is a result: outputs go to runs_dry/ and results/dry_run/.
source "$(dirname "$0")/common.sh"
MODEL="${MODEL:-real}"
STEPS="${STEPS:-6}"
R="${RUNS_DIR:-runs_dry}"
OUTD="${EVAL_OUT:-results/dry_run}"
CK=$(printf "ckpt_%07d" "$STEPS")
COMMON=(train.max_steps=$STEPS train.save_every=$STEPS train.keep_every=$STEPS train.val_every=$STEPS train.val_batches=2
        train.log_every=2 train.warmup_steps=2 train.sim_eval_every=$STEPS train.sim_eval_episodes=1 "train.sim_eval_tasks=[0]"
        eval.max_steps=6 eval.num_parallel_envs=1 train.system_monitor_interval_s=5 data.file_limit=2 train.batch_size=2
        train.grad_accum=1)
LORA=()
if [ "$MODEL" = "tiny" ]; then
  COMMON+=(monet.model_path=tiny-random monet.dtype=float32 train.mixed_precision=no prompt.image_size=112 data.num_workers=0)
  LORA=("monet.lora_layers=[0,1]")
  export MUJOCO_GL="${MUJOCO_GL_DRY:-osmesa}" PYOPENGL_PLATFORM="${MUJOCO_GL_DRY:-osmesa}"
  [ "$DEVICE" = "cuda" ] && ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" && DEVICE=cpu
else
  COMMON+=(data.num_workers=4)
fi
rm -rf "$R" "$OUTD"
t() { python -m monet_action.train --config "configs/$1.yaml" --device "$DEVICE" --override train.output_dir="$R/$1" "${COMMON[@]}" "${@:2}"; }
t d0_k0
t d1_k10
t d4_query
t d5_mlp
t d2_lora "${LORA[@]}" train.init_from="$R/d1_k10/$CK"
t d0_k0_lora "${LORA[@]}" train.init_from="$R/d0_k0/$CK"
python - "$R" <<'PY'
import json, sys, pathlib
bad = []
for d in sorted(pathlib.Path(sys.argv[1]).iterdir()):
    st = json.loads((d / "status.json").read_text())
    ev = [json.loads(l)["event"] for l in (d / "events.jsonl").read_text().splitlines()]
    ok = st["state"] == "finished" and "checkpoint" in ev and "gradient_audit_passed" in ev
    sim = "sim_eval" in ev
    print(f"{d.name:14s} state={st['state']:9s} step={st['step']} sim_eval={'ok' if sim else 'FAILED/absent'} events={len(ev)}")
    bad += [] if ok else [d.name]
sys.exit(1 if bad else 0)
PY
RUNS_DIR="$R" EVAL_OUT="$OUTD" REPORT_OUT="$OUTD/phase1_report.md" \
  EVAL_ARGS="--task-ids 0 --seeds 0 --episodes-per-task 1 --max-steps 6 --num-parallel-envs 1 --save-videos 1" \
  OFFLINE_ARGS="--max-batches 2" LATENCY_ARGS="--n-calls 3" \
  bash scripts/evaluate_all.sh
echo "DRY RUN OK: $(ls "$OUTD" | wc -l) evaluation dirs, report at $OUTD/phase1_report.md"
