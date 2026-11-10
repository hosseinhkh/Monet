#!/usr/bin/env bash
# G1: overfit 32 demonstration windows with the REAL Monet-7B (K=10, frozen) and check the
# deterministic overfit curve, then a 2-task closed-loop smoke rollout.
source "$(dirname "$0")/common.sh"
python -m monet_action.train --config configs/smoke.yaml --device "$DEVICE" \
  --override train.overfit_eval_every=50 "$@"
python - <<'PY'
import json
ov=[json.loads(l) for l in open("runs/smoke/overfit_log.jsonl")]
print("overfit fm_loss:", [round(r["fm_loss"],4) for r in ov])
print("overfit action_mse:", [round(r["action_mse"],4) for r in ov])
PY
python -m monet_action.evaluate closed_loop --checkpoint runs/smoke --eval-config configs/eval_smoke.yaml \
  --output-dir results/smoke/closed_loop --device "$DEVICE"
