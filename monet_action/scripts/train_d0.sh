#!/usr/bin/env bash
# D0: see configs/d0_k0.yaml. Re-running resumes from runs/d0_k0/last (deterministic).
source "$(dirname "$0")/common.sh"
python -m monet_action.train --config configs/d0_k0.yaml --device "$DEVICE" "$@"
