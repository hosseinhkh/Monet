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


def test_memory_check_separates_crash_from_oom(tmp_path, capsys):
    p = tmp_path / "memory_profile.json"
    p.write_text(json.dumps([{"batch_size": 2, "ok": True, "peak_gb": 30.0, "step_time_s": 1.0, "oom": False, "error": None},
                             {"batch_size": 4, "ok": False, "peak_gb": None, "step_time_s": None, "oom": True,
                              "error": "torch.OutOfMemoryError: CUDA out of memory"}]))
    assert preflight.check_memory(str(p), budget_gb=44) == 0
    assert "out of memory" in capsys.readouterr().out
    p.write_text(json.dumps([{"batch_size": 2, "ok": False, "peak_gb": None, "step_time_s": None, "oom": False,
                              "error": "TypeError: boom"}]))
    assert preflight.check_memory(str(p), budget_gb=44) == 1
    out = capsys.readouterr().out
    assert "CRASHED: TypeError: boom" in out and "nothing fits" not in out


def test_profile_memory_stops_at_a_crash(tmp_path):
    """A probe that fails for a non-memory reason is recorded as a crash and stops the profile."""
    import subprocess

    from conftest import ROOT, common_sh_env

    out = tmp_path / "mp.json"
    env = {**common_sh_env(), "BATCHES": "2 4", "PROFILE_OUT": str(out), "DEVICE": "cpu", "MONET_GPU_HOLD": "0",
           "PROFILE_OVERRIDES": "no_such_section.no_such_key=1"}
    r = subprocess.run(["bash", "scripts/profile_memory.sh", "configs/tiny_cpu.yaml"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=300)
    rows = json.loads(out.read_text())
    assert [x["batch_size"] for x in rows] == [2], r.stdout + r.stderr  # batch 4 not attempted
    assert rows[0]["ok"] is False and rows[0]["oom"] is False and rows[0]["error"]
    assert "CRASHED (not out of memory)" in r.stderr
    assert preflight.check_memory(str(out), budget_gb=44) == 1


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


def test_gpu_reserve_arithmetic(monkeypatch):
    """MONET_GPU_RESERVE: hold fraction*total minus what is already held, never more than free - margin."""
    import types

    import torch

    from monet_action import gpu

    G = gpu.GiB
    asked = {}
    fake = types.SimpleNamespace(is_available=lambda: True, mem_get_info=lambda i=0: (90 * G, 96 * G),
                                 memory_reserved=lambda i=0: 20 * G, OutOfMemoryError=MemoryError)
    monkeypatch.setattr(gpu, "torch", types.SimpleNamespace(
        cuda=fake, uint8=torch.uint8, empty=lambda n, dtype=None, device=None: asked.setdefault("n", n)))
    assert abs(gpu.reserve_gpu_memory(0.95, margin_gb=3) - 71.2) < 0.01  # 0.95*96 - 20
    asked.clear()
    assert abs(gpu.reserve_gpu_memory(1.0, margin_gb=10) - 76.0) < 0.01  # capped by free(90) - margin(10)
    assert gpu.reserve_gpu_memory(0.0) == 0.0


def test_common_sh_pins_one_gpu():
    import subprocess

    from conftest import ROOT, common_sh_env

    out = subprocess.run(["bash", "-c", "unset CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID; GPU_ID=3 MONET_GPU_HOLD=0 "
                          "source scripts/common.sh && echo $CUDA_VISIBLE_DEVICES $MUJOCO_EGL_DEVICE_ID $CUDA_DEVICE_ORDER"],
                         cwd=ROOT, capture_output=True, text=True, env=common_sh_env())
    assert out.stdout.split() == ["3", "3", "PCI_BUS_ID"], out.stderr


def _fake_nvidia_smi(tmp_path, monkeypatch, rows):
    """Put a fake `nvidia-smi` on PATH that prints `index, name, free MiB, total MiB` rows."""
    exe = tmp_path / "nvidia-smi"
    exe.write_text("#!/bin/sh\ncat <<'X'\n" + "\n".join(rows) + "\nX\n")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{__import__('os').environ['PATH']}")


def test_gpu_auto_picks_card_with_most_free_memory(tmp_path, monkeypatch):
    from monet_action import runtime_env

    _fake_nvidia_smi(tmp_path, monkeypatch, ["0, RTX PRO 6000, 2048, 97887", "1, RTX PRO 6000, 96000, 97887"])
    gpus = runtime_env.nvidia_smi_gpus()
    assert [g[0] for g in gpus] == [0, 1] and round(gpus[1][2]) == 94
    assert runtime_env.pick_gpu("auto") == "1"
    assert runtime_env.pick_gpu("0") == "0" and runtime_env.pick_gpu(None) == "0"


def test_apply_defaults_pins_gpu_and_respects_existing(monkeypatch, tmp_path):
    from monet_action import runtime_env

    for k in ("CUDA_VISIBLE_DEVICES", "MUJOCO_EGL_DEVICE_ID", "CUDA_DEVICE_ORDER", "MUJOCO_GL", "PYOPENGL_PLATFORM"):
        monkeypatch.delenv(k, raising=False)
    _fake_nvidia_smi(tmp_path, monkeypatch, ["0, A, 1000, 97887", "1, B, 90000, 97887"])
    monkeypatch.setenv("GPU_ID", "auto")
    runtime_env.apply_defaults()
    env = __import__("os").environ
    assert (env["CUDA_VISIBLE_DEVICES"], env["MUJOCO_EGL_DEVICE_ID"], env["CUDA_DEVICE_ORDER"]) == ("1", "1", "PCI_BUS_ID")
    assert env["MUJOCO_GL"] == "egl"  # an NVIDIA driver is present
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID")
    runtime_env.apply_defaults()
    assert env["CUDA_VISIBLE_DEVICES"] == "3" and env["MUJOCO_EGL_DEVICE_ID"] == "3"  # explicit choice wins


def test_reserve_respects_cap_and_shared_card(monkeypatch):
    """50% share of a 96 GB card with ~48 GB used by others: hold just under the cap, never beyond free."""
    import types

    import torch

    from monet_action import gpu

    G = gpu.GiB
    asked = {}
    fake = types.SimpleNamespace(is_available=lambda: True, mem_get_info=lambda i=0: (31 * G, 96 * G),
                                 memory_reserved=lambda i=0: 17 * G, OutOfMemoryError=MemoryError)
    monkeypatch.setattr(gpu, "torch", types.SimpleNamespace(
        cuda=fake, uint8=torch.uint8, empty=lambda n, dtype=None, device=None: asked.setdefault("n", n)))
    # target 0.5*96 - 1 (capped) = 47; minus 17 held = 30; free 31 - margin 3 = 28 -> 28 GB more (45 GB total)
    assert abs(gpu.reserve_gpu_memory(0.5, margin_gb=3, capped=True) - 28.0) < 0.01


def test_limit_fraction_defaults_to_reserve(monkeypatch):
    from monet_action import gpu

    monkeypatch.delenv("MONET_GPU_LIMIT", raising=False)
    monkeypatch.setenv("MONET_GPU_RESERVE", "0.5")
    assert gpu.limit_fraction() == 0.5
    monkeypatch.setenv("MONET_GPU_LIMIT", "0.6")
    assert gpu.limit_fraction() == 0.6
    monkeypatch.setenv("MONET_GPU_LIMIT", "0")
    assert gpu.limit_fraction() == 0.0
    monkeypatch.delenv("MONET_GPU_LIMIT")
    monkeypatch.setenv("MONET_GPU_RESERVE", "0")
    assert gpu.limit_fraction() == 0.0 and preflight.default_memory_budget_gb() == 85.0
    monkeypatch.setenv("MONET_GPU_RESERVE", "0.5")
    assert 40 < preflight.default_memory_budget_gb() < 47  # ~0.5 x card - 3 GB


def test_env_local_file(tmp_path, monkeypatch):
    from monet_action import runtime_env

    f = tmp_path / "env.local"
    f.write_text("# comment\nGPU_ID=1   # inline\nexport MONET_GPU_RESERVE=0.5\nHF_HOME=\"/data/hf\"\nnot a setting\n")
    assert runtime_env.read_env_local(f) == {"GPU_ID": "1", "MONET_GPU_RESERVE": "0.5", "HF_HOME": "/data/hf"}
