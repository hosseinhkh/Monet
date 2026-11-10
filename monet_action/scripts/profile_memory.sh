#!/usr/bin/env bash
# Measure peak GPU memory and step time for candidate micro-batch sizes (5 real training steps each).
#   bash scripts/profile_memory.sh [config] ; BATCHES="4 8 12 16" overrides the candidates.
# Writes results/preflight/memory_profile.json; `python -m monet_action.preflight memory` picks the batch.
source "$(dirname "$0")/common.sh"
CFG="${1:-configs/d2_lora.yaml}"
OUTJ="${PROFILE_OUT:-results/preflight/memory_profile.json}"
mkdir -p "$(dirname "$OUTJ")"
echo "[]" > "$OUTJ"
for B in ${BATCHES:-4 8 12 16}; do
  D="runs/memprobe_b$B"
  rm -rf "$D"
  ok=1
  python -m monet_action.train --config "$CFG" --device "$DEVICE" --override \
    train.output_dir=$D train.max_steps=5 train.batch_size=$B train.grad_accum=1 train.log_every=1 \
    train.val_every=1000000 train.save_every=1000000 train.sim_eval_every=0 train.init_from=null \
    data.num_workers=4 ${PROFILE_OVERRIDES:-} || ok=0
  python - "$D" "$B" "$ok" "$OUTJ" <<'PY'
import json, sys, pathlib
d, b, ok, out = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4]
mp = json.loads((d / "memory_profile.json").read_text()) if (d / "memory_profile.json").exists() else {}
log = [json.loads(l) for l in (d / "train_log.jsonl").read_text().splitlines()] if (d / "train_log.jsonl").exists() else []
steps = [r["step_time_s"] for r in log if "step_time_s" in r][2:]  # skip warm-up steps
row = {"batch_size": b, "ok": ok, "peak_gb": round(mp["peak_gb_after_3_steps"], 2) if mp.get("peak_gb_after_3_steps") == mp.get("peak_gb_after_3_steps") and mp else None,
       "step_time_s": round(sum(steps) / len(steps), 3) if steps else None}
rows = json.loads(pathlib.Path(out).read_text()); rows.append(row)
pathlib.Path(out).write_text(json.dumps(rows, indent=2)); print("[profile]", row)
PY
  rm -rf "$D"
done
