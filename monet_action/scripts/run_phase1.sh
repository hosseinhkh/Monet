#!/usr/bin/env bash
# Whole Phase 1 on one GPU: preflight -> training -> evaluation -> report. Every stage resumes.
source "$(dirname "$0")/common.sh"
bash scripts/preflight.sh
bash scripts/train_all.sh
bash scripts/evaluate_all.sh
