# results/

**From this repository's CPU container (no GPU).** "Tiny" means Monet's real modeling code with a tiny
random-weight configuration, so it validates mechanisms, not Monet-7B.

* `audit/g0_report_tiny.{json,md}`: the G0 audit on real LeRobot LIBERO-Spatial data and the real LIBERO
  simulator (OSMesa), with Monet's real modeling code in the tiny configuration. It covers both latent
  layouts, parity errors, K=0 vs K=10, state dependence, data conventions and simulator I/O.
* `audit/*.png`: dataset frame vs simulator frame (policy view, rotated 180°) for the orientation check.
* `cpu_validation/pytest_all.txt`: the full test-suite log.
* `cpu_validation/overfit_tiny/`: the G1 overfit on 32 real demonstration windows with the tiny Monet,
  in the same layout a GPU run produces:
  * `overfit_log.jsonl`: the deterministic, monotone curve;
  * `train_log.jsonl`, `status.json`, `events.jsonl`, `system.jsonl` and `logs/train.log`;
  * `representation.json` and the gradient audit.
* `cpu_validation/closed_loop_tiny/`: two real LIBERO episodes on development init states, driven
  through CALL_ACTION / TOOL_RESULT by the tiny overfit policy (24-step cap, E=2). It holds the traces,
  the per-call latency split, failure tags, one video, and the eval status and logs.
  **0/2 success is expected and is not a result.**

**Produced on the GPU server (not present yet):**

* `audit/g0_report_monet7b.*`, `audit/monet7b_gpu_tests.json` and the pytest logs, from
  `scripts/audit_upstream.sh`;
* `eval/<run>/…` and `phase1_report.md`, from `scripts/evaluate_all.sh`.
