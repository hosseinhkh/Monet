"""GPU holder <-> lease protocol (monet_action.gpu_hold) with a fake allocator: no GPU needed.

The holder runs in a thread of this test process; clients are real subprocesses, so lease files, pid
liveness, atexit cleanup and crash (SIGKILL) recovery are exercised exactly as on the server."""
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from monet_action import gpu_hold

from conftest import ROOT

G = gpu_hold.GiB
CLIENT = ("import sys; from pathlib import Path; from monet_action.gpu_hold import acquire_lease; "
          "ok = acquire_lease(timeout_s=20, d=Path(sys.argv[1])); print('LEASED', ok, flush=True); sys.stdin.read()")


class FakeCard:
    """96 GB card, 40 GB used by someone else; the holder's chunks count while referenced."""

    def __init__(self, total=96 * G, others=40 * G):
        self.total, self.others, self.held = total, others, 0
        self.lock = threading.Lock()

    def alloc(self, n):
        card = self

        class Chunk:
            nbytes = n

            def __del__(self):
                with card.lock:
                    card.held -= n

        with self.lock:
            if n > self.total - self.others - self.held:
                raise MemoryError("fake OOM")
            self.held += n
        return Chunk()

    def mem_info(self):
        with self.lock:
            return self.total - self.others - self.held, self.total


def _wait(pred, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture()
def holder(tmp_path):
    card, stop = FakeCard(), threading.Event()
    t = threading.Thread(target=gpu_hold.run_holder, kwargs=dict(
        fraction=0.5, margin_gb=2.0, poll_s=0.02, d=tmp_path, alloc=card.alloc, release=lambda: None,
        mem_info=card.mem_info, should_stop=stop.is_set), daemon=True)
    t.start()
    assert _wait(lambda: gpu_hold.read_state(tmp_path).get("state") == "holding")
    yield card, tmp_path
    stop.set()
    t.join(5)


def _client(d):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    p = subprocess.Popen([sys.executable, "-c", CLIENT, str(d)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, env=env)
    assert p.stdout.readline().strip() == "LEASED True"
    return p


def test_no_holder_means_no_lease(tmp_path):
    assert gpu_hold.acquire_lease(timeout_s=1, d=tmp_path) is False
    assert not list(tmp_path.glob("lease.*"))


def test_holder_claims_its_share_but_leaves_a_margin(holder):
    card, d = holder
    assert card.held == 46 * G  # 0.5 * 96 - 2 GB margin
    st = gpu_hold.read_state(d)
    assert st["state"] == "holding" and abs(st["held_gb"] - 46) < 1e-6 and gpu_hold.holder_pid(d) == os.getpid()


def test_hand_over_to_a_client_and_back(holder):
    card, d = holder
    p = _client(d)
    # the client only returns from acquire_lease once the holder has seen *its* lease and freed the memory
    assert card.held == 0
    st = gpu_hold.read_state(d)
    assert st["state"] == "released" and p.pid in st["leases"]
    p.stdin.close()  # normal exit: atexit removes the lease, the holder takes the share back
    p.wait(10)
    assert _wait(lambda: card.held == 46 * G and gpu_hold.read_state(d).get("state") == "holding")
    assert not list(d.glob("lease.*"))


def test_crashed_client_does_not_strand_the_share(holder):
    card, d = holder
    p = _client(d)
    assert card.held == 0
    os.kill(p.pid, signal.SIGKILL)  # no atexit: the stale lease file stays behind
    p.wait(10)
    assert _wait(lambda: card.held == 46 * G)
    assert not list(d.glob("lease.*"))  # the holder removed the dead process's lease


def test_holder_grows_as_others_free_memory(tmp_path):
    card, stop = FakeCard(others=80 * G), threading.Event()
    t = threading.Thread(target=gpu_hold.run_holder, kwargs=dict(
        fraction=0.5, margin_gb=2.0, poll_s=0.02, d=tmp_path, alloc=card.alloc, release=lambda: None,
        mem_info=card.mem_info, should_stop=stop.is_set), daemon=True)
    t.start()
    try:
        assert _wait(lambda: gpu_hold.read_state(tmp_path).get("state") == "growing")
        assert card.held == 14 * G  # only 16 GB free: takes all but the margin
        card.others = 20 * G  # the other user finishes
        assert _wait(lambda: gpu_hold.read_state(tmp_path).get("state") == "holding")
        assert card.held == 46 * G
    finally:
        stop.set()
        t.join(5)
    assert gpu_hold.read_state(tmp_path)["state"] == "stopped" and gpu_hold.holder_pid(tmp_path) is None
    assert card.held == 0


def test_stale_pid_files_are_ignored(tmp_path):
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    (tmp_path / "holder.pid").write_text(f"{p.pid} 1")
    (tmp_path / f"lease.{p.pid}").write_text("1")
    assert gpu_hold.holder_pid(tmp_path) is None
    assert gpu_hold.live_leases(tmp_path) == [] and not (tmp_path / f"lease.{p.pid}").exists()
    # a live pid with a different start time is a reused pid, not the original process
    (tmp_path / "holder.pid").write_text(f"{os.getpid()} 1")
    assert gpu_hold.holder_pid(tmp_path) is (None if gpu_hold._start_time(os.getpid()) else os.getpid())


def test_cli_status_and_stop_without_holder(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MONET_GPU_HOLD_DIR", str(tmp_path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    gpu_hold.main(["status"])
    assert '"running": false' in capsys.readouterr().out
    gpu_hold.main(["stop"])
    assert "not running" in capsys.readouterr().out
    monkeypatch.setenv("MONET_GPU_RESERVE", "0")
    with pytest.raises(SystemExit):
        gpu_hold.main(["start"])


def test_setup_gpu_takes_the_lease_before_touching_memory(monkeypatch):
    """setup_gpu is the single entry point train/evaluate/audit call before loading anything."""
    from monet_action import gpu

    calls = []
    monkeypatch.setattr(gpu_hold, "acquire_lease", lambda *a, **k: calls.append("lease") or True)

    class Cuda:
        OutOfMemoryError = MemoryError

        @staticmethod
        def is_available():
            return True

        def __getattr__(self, name):
            def f(*a, **k):
                calls.append(name)
                return {"mem_get_info": (40 * G, 96 * G), "memory_reserved": 0, "device_count": 1}.get(name)
            return f

    cuda = Cuda()
    props = type("P", (), {"name": "fake", "major": 12, "minor": 0, "total_memory": 96 * G})()
    cuda.get_device_properties = lambda i: props
    fake = type("T", (), {"cuda": cuda, "backends": gpu.torch.backends, "uint8": None,
                          "empty": lambda *a, **k: calls.append("empty")})
    monkeypatch.setattr(gpu, "torch", fake)
    monkeypatch.setenv("MONET_GPU_RESERVE", "0.5")
    s = gpu.setup_gpu("cuda")
    assert calls[0] == "lease" and "empty" in calls and s["holder_lease"] is True
    assert gpu.setup_gpu("cpu") == {"cuda": False}


def _env(tmp_path, **kw):
    return {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "MONET_GPU_HOLD_DIR": str(tmp_path), "MONET_GPU_HOLD_FAKE_GB": "96", "CUDA_VISIBLE_DEVICES": "5", **kw}


def _cleanup(d):
    pid = gpu_hold.holder_pid(d)
    if pid is not None:
        os.kill(pid, signal.SIGTERM)
        _wait(lambda: gpu_hold.holder_pid(d) is None)


def test_cli_start_hand_over_stop(tmp_path):
    """The real CLI: detached holder process, a client lease, stop. Fake card: no GPU memory is touched."""
    env, d = _env(tmp_path), tmp_path / "gpu5"
    run = lambda *a: subprocess.run([sys.executable, "-m", "monet_action.gpu_hold", *a], env=env, cwd=ROOT,  # noqa: E731
                                    capture_output=True, text=True, timeout=60)
    try:
        out = run("start", "--fraction", "0.5", "--poll", "0.05")
        assert out.returncode == 0 and "holding" in out.stdout, out.stdout + out.stderr
        st = gpu_hold.read_state(d)
        assert st["state"] == "holding" and abs(st["held_gb"] - 46) < 1e-6
        hp = gpu_hold.holder_pid(d)
        assert hp is not None and hp != os.getpid()
        assert "already running" in run("start").stdout and run("ensure").stdout == ""
        p = subprocess.Popen([sys.executable, "-c", CLIENT, str(d)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             text=True, env=env)
        assert p.stdout.readline().strip() == "LEASED True"
        assert gpu_hold.read_state(d)["state"] == "released"
        p.stdin.close()
        p.wait(10)
        assert _wait(lambda: gpu_hold.read_state(d).get("state") == "holding")
        out = run("stop")
        assert "stopped" in out.stdout and gpu_hold.holder_pid(d) is None
        assert gpu_hold.read_state(d)["state"] == "stopped"
    finally:
        _cleanup(d)


def test_common_sh_starts_holder_only_when_asked(tmp_path):
    venv = os.environ.get("VENV") or os.path.dirname(os.path.dirname(sys.executable))
    d = tmp_path / "gpu5"
    sh = lambda extra: subprocess.run(["bash", "-c", f"export VENV={venv}; {extra} source scripts/common.sh"],  # noqa: E731
                                      env=_env(tmp_path, MONET_GPU_RESERVE="0.5"), cwd=ROOT, capture_output=True,
                                      text=True, timeout=90)
    try:
        sh("MONET_GPU_HOLD=0")
        sh("MONET_GPU_HOLD=1 DEVICE=cpu")
        assert gpu_hold.holder_pid(d) is None
        out = sh("MONET_GPU_HOLD=1 DEVICE=cuda")
        assert out.returncode == 0, out.stderr
        assert gpu_hold.holder_pid(d) is not None and gpu_hold.read_state(d)["state"] == "holding"
        out = subprocess.run(["bash", "scripts/gpu_hold.sh", "stop"], env=_env(tmp_path, VENV=venv, MONET_GPU_HOLD="1"),
                             cwd=ROOT, capture_output=True, text=True, timeout=60)
        assert "stopped" in out.stdout, out.stdout + out.stderr
        assert gpu_hold.holder_pid(d) is None
    finally:
        _cleanup(d)
