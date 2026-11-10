#!/usr/bin/env bash
# D4: see configs/d4_query.yaml. Re-running resumes from runs/d4_query/last (deterministic).
source "$(dirname "$0")/common.sh"
python -m monet_action.train --config configs/d4_query.yaml --device "$DEVICE" "$@"
