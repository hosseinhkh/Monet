#!/usr/bin/env bash
# D2: see configs/d2_lora.yaml. Re-running resumes from runs/d2_lora/last (deterministic).
source "$(dirname "$0")/common.sh"
python -m monet_action.train --config configs/d2_lora.yaml --device "$DEVICE" "$@"
