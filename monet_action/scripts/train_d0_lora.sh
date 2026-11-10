#!/usr/bin/env bash
# D0-LoRA: K=0 + LoRA, stage B from runs/d0_k0/ckpt_0010000 (matched control for D2). See configs/d0_k0_lora.yaml.
source "$(dirname "$0")/common.sh"
python -m monet_action.train --config configs/d0_k0_lora.yaml --device "$DEVICE" "$@"
