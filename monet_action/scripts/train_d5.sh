#!/usr/bin/env bash
# D5: see configs/d5_mlp.yaml. Re-running resumes from runs/d5_mlp/last (deterministic).
source "$(dirname "$0")/common.sh"
python -m monet_action.train --config configs/d5_mlp.yaml --device "$DEVICE" "$@"
