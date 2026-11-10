#!/usr/bin/env bash
# Original LIBERO-Spatial demonstrations (hdf5, with recorded MuJoCo states). Used only by the
# simulator test that replays a recorded demo end to end (tests/test_libero_sim.py); training uses
# the LeRobot parquet data. Default: the task-0 file (~500 MB). ALL=1 downloads all 10 (~6.2 GB).
source "$(dirname "$0")/common.sh"
OUT="${OUT:-$HERE/data/libero_raw}"
python - <<PY
from huggingface_hub import hf_hub_download, snapshot_download
rev = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
if "${ALL:-0}" == "1":
    snapshot_download("yifengzhu-hf/LIBERO-datasets", repo_type="dataset", revision=rev,
                      allow_patterns=["libero_spatial/*"], local_dir="$OUT")
else:
    print(hf_hub_download("yifengzhu-hf/LIBERO-datasets", repo_type="dataset", revision=rev, local_dir="$OUT",
          filename="libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"))
PY
