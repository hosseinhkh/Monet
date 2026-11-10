#!/usr/bin/env bash
# Monitoring dashboard for training runs and evaluations (stdlib only; reads runs/ and results/).
#   bash scripts/dashboard.sh                         # http://localhost:8765 (bound to 127.0.0.1)
#   bash scripts/dashboard.sh --port 9000 --refresh 5
#   MONET_DASHBOARD_TOKEN=$(openssl rand -hex 16) bash scripts/dashboard.sh --host 0.0.0.0   # LAN + token
# From a laptop:  ssh -L 8765:localhost:8765 <user>@<gpu-server>   then open http://localhost:8765
# Debug bundles without the UI:  python -m monet_action.bundle runs/<run> results/eval/<run> -o debug.zip
source "$(dirname "$0")/common.sh"
python -m monet_action.dashboard --runs "${RUNS_DIR:-runs}" --results "${RESULTS_DIR:-results}" "$@"
