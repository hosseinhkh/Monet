#!/usr/bin/env bash
# LeRobot LIBERO-Spatial (432 demos, ~6.5 GB parquet) + Monet-7B weights (~16.6 GB), pinned revisions.
source "$(dirname "$0")/common.sh"
DATA_DIR="${DATA_DIR:-$HERE/data}"
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("lerobot/libero_spatial_image", repo_type="dataset",
                  revision="d86c0b94922572b3b657e1d1a3d01f0952ddeb46",
                  local_dir="$DATA_DIR/libero_spatial_image")
if "${SKIP_MONET:-0}" != "1":
    p = snapshot_download("NOVAglow646/Monet-7B", revision="39c1e320a3e15c357bff911fda263923edbdda37")
    print("Monet-7B at", p)
PY
