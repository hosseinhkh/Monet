# results/

* `audit/g0_report_tiny.{json,md}`: G0 audit run in the CPU container. It uses real LeRobot LIBERO-Spatial
  data, the real LIBERO simulator (OSMesa) and Monet's real modeling code with a **tiny random** config, so
  it validates mechanisms only.
* `audit/*.png`: dataset frame vs simulator frame (policy view, rotated 180°), used for the orientation
  check.
* `cpu_validation/pytest_all.txt`: full test-suite log (51 tests).
* `cpu_validation/overfit_tiny/`: logs of the G1 overfit on 32 real demonstration windows with the tiny
  Monet. `overfit_log.jsonl` holds the deterministic, monotone curve.
* `cpu_validation/closed_loop_tiny/`: two real LIBERO episodes driven through CALL_ACTION/TOOL_RESULT with
  the tiny overfit policy (24-step cap, E=2): traces, per-call latency split and one video.
  **0/2 success is expected and is not a result.**
* `audit/g0_report_monet7b.*`, `eval/`, `phase1_report.md`: **produced on the GPU server** by
  `scripts/audit_upstream.sh` and `scripts/evaluate_all.sh`. Not present yet.
