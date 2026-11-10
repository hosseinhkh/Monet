"""Run monitoring: status heartbeat, events, text logs, system metrics, failure capture."""
import json
import time

import pytest

from monet_action.logging_utils import RunMonitor, system_snapshot


def _jsonl(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_monitor_lifecycle(tmp_path):
    d = tmp_path / "run"
    with RunMonitor(d, kind="train", name="x", variant="D1", max_steps=10, system_interval_s=0.05) as mon:
        mon.logger.info("hello from the run")
        mon.update(step=3, last_loss=0.5)
        mon.event("checkpoint", step=3, path="ckpt_0000003")
        st = json.loads((d / "status.json").read_text())
        assert st["state"] == "running" and st["step"] == 3 and st["max_steps"] == 10 and st["pid"]
        time.sleep(0.2)
    st = json.loads((d / "status.json").read_text())
    assert st["state"] == "finished" and st["error"] is None and st["updated_at"] >= st["started_at"]
    ev = [e["event"] for e in _jsonl(d / "events.jsonl")]
    assert ev[0] == "start" and "checkpoint" in ev and ev[-1] == "finished"
    assert "hello from the run" in (d / "logs" / "train.log").read_text()
    sysrec = _jsonl(d / "system.jsonl")
    assert sysrec and {"time", "gpus", "cpu_percent", "ram_used_gb", "disk_free_gb"} <= set(sysrec[0])


def test_monitor_records_failures_and_reraises(tmp_path):
    d = tmp_path / "bad"
    with pytest.raises(ZeroDivisionError):
        with RunMonitor(d, kind="eval", system_interval_s=0):
            _ = 1 / 0
    st = json.loads((d / "status.json").read_text())
    assert st["state"] == "failed" and "ZeroDivisionError" in st["error"]
    assert "ZeroDivisionError" in (d / "logs" / "eval.log").read_text()
    assert _jsonl(d / "events.jsonl")[-1]["event"] == "failed"


def test_keyboard_interrupt_marks_stopped(tmp_path):
    d = tmp_path / "stop"
    with pytest.raises(KeyboardInterrupt):
        with RunMonitor(d, system_interval_s=0):
            raise KeyboardInterrupt
    assert json.loads((d / "status.json").read_text())["state"] == "stopped"


def test_restart_keeps_first_start_time(tmp_path):
    d = tmp_path / "r"
    with RunMonitor(d, system_interval_s=0):
        pass
    first = json.loads((d / "status.json").read_text())["started_at"]
    time.sleep(0.01)
    with RunMonitor(d, system_interval_s=0):
        pass
    st = json.loads((d / "status.json").read_text())
    assert st["first_started_at"] == first and st["started_at"] > first


def test_system_snapshot_fields(tmp_path):
    s = system_snapshot(str(tmp_path))
    assert isinstance(s["gpus"], list) and s["disk_free_gb"] > 0 and s["ram_total_gb"] > 0


@pytest.mark.slow
@pytest.mark.tokenizer
def test_training_run_writes_dashboard_contract(cfg_path, data_root, processor, tmp_path):
    from monet_action.config import load_config
    from monet_action.train import train

    out = tmp_path / "run"
    cfg = load_config(cfg_path, [f"data.root={data_root}", f"train.output_dir={out}", "train.max_steps=4",
                                 "train.log_every=2", "train.save_every=2", "train.system_monitor_interval_s=0.05",
                                 "data.max_windows=16"])
    train(cfg, device="cpu")
    st = json.loads((out / "status.json").read_text())
    assert st["state"] == "finished" and st["step"] == 4 and st["kind"] == "train" and st["variant"] == cfg.variant
    ev = [e["event"] for e in _jsonl(out / "events.jsonl")]
    assert {"start", "gradient_audit_passed", "checkpoint", "finished"} <= set(ev)
    log = (out / "logs" / "train.log").read_text()
    assert "step 4/4" in log and "representation" in log
    rows = [r for r in _jsonl(out / "train_log.jsonl") if "loss" in r]
    assert rows and {"step", "loss", "time", "lrs", "step_time_s", "grad_norm"} <= set(rows[-1])
    rep = json.loads((out / "representation.json").read_text())
    assert rep["num_latents_K"] == cfg.monet.num_latents and rep["latent_layer"] == -1 and "example_prompt" in rep
