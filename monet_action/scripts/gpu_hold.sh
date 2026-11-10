#!/usr/bin/env bash
# Keep your share of the GPU (MONET_GPU_RESERVE, e.g. 0.5 of GPU_ID) claimed between runs.
#   bash scripts/gpu_hold.sh start [--fraction 0.5]   # background holder; survives logout / tmux detach
#   bash scripts/gpu_hold.sh status                   # held GB, live leases (Monet-Action processes using the share)
#   bash scripts/gpu_hold.sh stop                     # give the memory back to the other users of the card
# Training / evaluation / tests take the share over from the holder automatically and hand it back on exit.
# With MONET_GPU_HOLD=1 in env.local, every script that sources common.sh starts the holder if needed.
MONET_GPU_HOLD_AUTOSTART=0 source "$(dirname "$0")/common.sh"
python -m monet_action.gpu_hold "${1:-status}" "${@:2}"
