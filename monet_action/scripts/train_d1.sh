#!/usr/bin/env bash
# D1: see configs/d1_k10.yaml. Re-running resumes from runs/d1_k10/last (deterministic).
source "$(dirname "$0")/common.sh"
python -m monet_action.train --config configs/d1_k10.yaml --device "$DEVICE" "$@"
