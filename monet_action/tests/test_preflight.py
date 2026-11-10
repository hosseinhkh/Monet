"""Preflight parsers that only run on the GPU server (memory profile, GPU-test log) + CPU-side checks."""
import json

from monet_action import preflight


def test_memory_recommendation(tmp_path):
    p = tmp_path / "memory_profile.json"
    p.write_text(json.dumps([
        {"batch_size": 4, "ok": True, "peak_gb": 40.1, "step_time_s": 0.6},
        {"batch_size": 8, "ok": True, "peak_gb": 61.0, "step_time_s": 1.0},
        {"batch_size": 12, "ok": True, "peak_gb": 88.2, "step_time_s": 1.4},
        {"batch_size": 16, "ok": False, "peak_gb": None, "step_time_s": None},
    ]))
    assert preflight.check_memory(str(p), budget_gb=85) == 0
    assert (tmp_path / "recommended_batch.txt").read_text().strip() == "train.batch_size=8 train.grad_accum=2"
    p.write_text(json.dumps([{"batch_size": 4, "ok": False, "peak_gb": None, "step_time_s": None}]))
    assert preflight.check_memory(str(p)) == 1


def test_gpu_test_log_must_not_skip(tmp_path):
    log = tmp_path / "gpu.log"
    log.write_text("........x\n8 passed, 1 xfailed in 300.1s\n")
    assert preflight.check_gpu_tests(str(log)) == 0
    log.write_text("SKIPPED [1] tests/test_monet7b_gpu.py:40: no CUDA device\n1 skipped in 0.1s\n")
    assert preflight.check_gpu_tests(str(log)) == 1
    log.write_text("1 failed, 7 passed in 200s\n")
    assert preflight.check_gpu_tests(str(log)) == 1


def test_overfit_check_on_synthetic_run(tmp_path):
    d = tmp_path / "smoke"
    d.mkdir()
    (d / "status.json").write_text(json.dumps({"state": "finished"}))
    (d / "gradient_audit.json").write_text(json.dumps({"passed": True, "report": {"action_head": {"grad_norm": 1.0}}}))
    rows = [{"fm_loss": v, "gripper_acc": 0.95} for v in (1.4, 0.8, 0.4, 0.3, 0.2)]
    (d / "overfit_log.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert preflight.check_overfit(str(d)) == 0
    rows = [{"fm_loss": v, "gripper_acc": 0.5} for v in (1.4, 1.5, 1.6, 1.5)]
    (d / "overfit_log.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert preflight.check_overfit(str(d)) == 1
