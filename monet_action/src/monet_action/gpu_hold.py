"""Keep a share of the GPU claimed *between* runs (after the preflight, between training variants, ...).

    bash scripts/gpu_hold.sh start [--fraction 0.5]   # background holder (default: MONET_GPU_RESERVE)
    bash scripts/gpu_hold.sh status
    bash scripts/gpu_hold.sh stop                     # give the share back to the other users of the card

(``python -m monet_action.gpu_hold <cmd>`` does the same.) With ``MONET_GPU_HOLD=1`` (env.local) every
script that sources scripts/common.sh starts the holder if it is not running yet, so after the preflight the
share stays claimed until you stop it.

Hand-over: the holder allocates ``fraction`` of the selected card and sleeps. Every Monet-Action process that
uses the GPU (training, evaluation, audit, preflight env check, Monet-7B GPU tests) first takes a *lease* in
:func:`monet_action.gpu.setup_gpu` - before it loads data or weights: it writes ``lease.<pid>``, waits until
the holder has released its memory (about one poll, 1 s), then claims the same share itself
(MONET_GPU_RESERVE / MONET_GPU_LIMIT). When the last lease's process exits - normally or by crashing - the
holder takes the memory back within ``--poll`` seconds. Other users of the card can therefore only get this
share during those short hand-overs, never while you are between steps.

State lives in ``$MONET_GPU_HOLD_DIR`` (default ``~/.cache/monet_action/gpu_hold``)/gpu<N>/ and is per GPU:
``holder.pid``, ``state.json`` (rewritten every poll), ``lease.<pid>`` and ``holder.log``. The holder keeps a
small CUDA context (~0.5 GB) while it has released the share to a lease.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

GiB = 1024 ** 3


# ------------------------------------------------------------------------------------------ files
def physical_gpu() -> str:
    return (os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("GPU_ID") or "0").split(",")[0].strip() or "0"


def hold_dir(gpu: Optional[str] = None) -> Path:
    base = Path(os.environ.get("MONET_GPU_HOLD_DIR", Path.home() / ".cache" / "monet_action" / "gpu_hold"))
    d = base / f"gpu{gpu or physical_gpu()}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _start_time(pid: int) -> str:
    """Kernel start time of ``pid`` ('' if unknown): tells a live process from a later one with a reused pid."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except Exception:
        return ""


def _alive(pid: int, start: str = "") -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return not start or not _start_time(pid) or _start_time(pid) == start


def read_state(d: Path) -> dict:
    try:
        return json.loads((d / "state.json").read_text())
    except Exception:
        return {}


def _write_state(d: Path, **st) -> None:
    tmp = d / f".state.{os.getpid()}.{threading.get_ident()}.tmp"
    tmp.write_text(json.dumps({**st, "updated_at": time.time()}))
    os.replace(tmp, d / "state.json")


def holder_pid(d: Path) -> Optional[int]:
    try:
        parts = (d / "holder.pid").read_text().split()
        pid, start = int(parts[0]), (parts[1] if len(parts) > 1 else "")
    except Exception:
        return None
    return pid if _alive(pid, start) else None


def live_leases(d: Path) -> List[int]:
    pids = []
    for f in d.glob("lease.*"):
        try:
            pid = int(f.name.split(".", 1)[1])
            start = f.read_text().strip()
        except (ValueError, OSError):
            continue
        if _alive(pid, start):
            pids.append(pid)
        else:
            f.unlink(missing_ok=True)  # crashed or finished without cleaning up
    return sorted(pids)


# ------------------------------------------------------------------------------------------ lease (clients)
def acquire_lease(timeout_s: float = 120.0, d: Optional[Path] = None) -> bool:
    """Ask a running holder for its share; call *before* this process allocates GPU memory.

    No-op (returns False) when no holder runs on this GPU. Otherwise writes ``lease.<pid>`` (removed when this
    process exits; a crashed process's lease is ignored by the holder) and waits until the holder reports that
    it has seen the lease and released its memory. Returns True once the memory is free (or after
    ``timeout_s`` with a warning)."""
    d = d or hold_dir()
    hp = holder_pid(d)
    me = os.getpid()
    if hp is None or hp == me:
        return False
    lease = d / f"lease.{me}"
    if not lease.exists():
        lease.write_text(_start_time(me))
        atexit.register(lambda: os.getpid() == me and lease.unlink(missing_ok=True))  # not in forked workers
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = read_state(d)
        if (st.get("state") == "released" and me in st.get("leases", [])) or holder_pid(d) is None:
            return True
        time.sleep(0.1)
    print(f"[gpu_hold] holder pid {hp} did not release within {timeout_s:.0f}s; continuing", file=sys.stderr)
    return True


def release_lease(d: Optional[Path] = None) -> None:
    """Give the share back to the holder before this process exits (optional: exit does it too)."""
    (d or hold_dir()).joinpath(f"lease.{os.getpid()}").unlink(missing_ok=True)


# ------------------------------------------------------------------------------------------ holder loop
def run_holder(fraction: float, margin_gb: float = 2.0, poll_s: float = 1.0, d: Optional[Path] = None,
               alloc: Optional[Callable[[int], object]] = None, release: Optional[Callable[[], None]] = None,
               mem_info: Optional[Callable[[], tuple]] = None, should_stop: Optional[Callable[[], bool]] = None) -> None:
    """Hold ``fraction`` of the card while no lease is live; release it while any lease is live.

    Never takes the last ``margin_gb`` of free memory. ``alloc``/``release``/``mem_info``/``should_stop`` are
    injectable for tests (default: torch on cuda:0, stop on SIGTERM / SIGINT)."""
    d = d or hold_dir()
    if alloc is None:
        import torch

        torch.cuda.set_device(0)
        alloc = lambda n: torch.empty(n, dtype=torch.uint8, device="cuda:0")  # noqa: E731
        release = torch.cuda.empty_cache
        mem_info = lambda: torch.cuda.mem_get_info(0)  # noqa: E731
    flag = {"stop": False}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: flag.update(stop=True))
    stop = should_stop or (lambda: flag["stop"])
    (d / "holder.pid").write_text(f"{os.getpid()} {_start_time(os.getpid())}")
    chunks: list = []
    held = lambda: sum(getattr(c, "nbytes", 0) for c in chunks)  # noqa: E731
    base = {"pid": os.getpid(), "fraction": fraction, "gpu": physical_gpu()}
    try:
        while not stop():
            free, total = mem_info()
            target = int(fraction * total - margin_gb * GiB)
            leases = live_leases(d)
            if leases:
                if chunks:
                    chunks.clear()
                    release()
                _write_state(d, state="released", held_gb=0.0, target_gb=target / GiB, leases=leases, **base)
            else:
                want = target - held()
                room = int(free - margin_gb * GiB)
                if want > GiB // 2 and room > GiB // 2:
                    try:
                        chunks.append(alloc(min(want, room)))
                    except Exception:
                        pass  # someone else got there first; retry next poll
                h = held()
                _write_state(d, state="holding" if h >= target - GiB else "growing", held_gb=h / GiB,
                             target_gb=target / GiB, leases=[], **base)
            time.sleep(poll_s)
    finally:
        chunks.clear()
        if release:
            release()
        _write_state(d, state="stopped", held_gb=0.0, leases=[], **base)
        (d / "holder.pid").unlink(missing_ok=True)


class _FakeCard:
    """CPU stand-in for the CLI tests (``MONET_GPU_HOLD_FAKE_GB=96``): counts bytes, allocates nothing."""

    def __init__(self, total_gb: float):
        self.total, self.held = int(total_gb * GiB), 0

    def alloc(self, n: int):
        self.held += n
        return _FakeChunk(self, n)

    def release(self) -> None:
        pass

    def mem_info(self) -> tuple:
        return self.total - self.held, self.total


class _FakeChunk:
    def __init__(self, card: _FakeCard, n: int):
        self.card, self.nbytes = card, n

    def __del__(self):
        self.card.held -= self.nbytes


# ------------------------------------------------------------------------------------------ CLI
def _status(d: Path) -> dict:
    pid = holder_pid(d)
    return {"gpu": physical_gpu(), "running": pid is not None, "holder_pid": pid, "leases": live_leases(d),
            "dir": str(d), **{k: v for k, v in read_state(d).items() if k not in ("pid", "leases", "gpu")}}


def main(argv=None) -> None:
    from .runtime_env import apply_defaults

    apply_defaults()  # same GPU pinning (GPU_ID -> CUDA_VISIBLE_DEVICES) and env.local as every other tool
    ap = argparse.ArgumentParser(prog="python -m monet_action.gpu_hold")
    ap.add_argument("cmd", choices=["start", "ensure", "status", "stop", "run"],
                    help="ensure = start quietly unless already running")
    ap.add_argument("--fraction", type=float, default=float(os.environ.get("MONET_GPU_RESERVE", "0") or 0))
    ap.add_argument("--margin-gb", type=float, default=float(os.environ.get("MONET_GPU_HOLD_MARGIN_GB", "2")))
    ap.add_argument("--poll", type=float, default=1.0)
    ap.add_argument("--wait", type=float, default=60.0, help="start: seconds to wait for the holder to claim memory")
    a = ap.parse_args(argv)
    d = hold_dir()
    pid = holder_pid(d)
    if a.cmd in ("start", "ensure"):
        if pid is not None:
            if a.cmd == "start":
                print(f"[gpu_hold] already running on GPU {physical_gpu()} (pid {pid}): {json.dumps(_status(d))}")
            return
        if not 0 < a.fraction < 1:
            sys.exit("[gpu_hold] set --fraction or MONET_GPU_RESERVE to a share between 0 and 1 (e.g. 0.5)")
        with open(d / "holder.log", "a") as log:
            p = subprocess.Popen([sys.executable, "-m", "monet_action.gpu_hold", "run", "--fraction", str(a.fraction),
                                  "--margin-gb", str(a.margin_gb), "--poll", str(a.poll)], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, start_new_session=True, env=os.environ.copy())
        t0 = time.time()
        while time.time() - t0 < a.wait:
            st = read_state(d)
            if st.get("pid") == p.pid and (st.get("state") in ("holding", "released")
                                           or (st.get("state") == "growing" and time.time() - t0 > 10)):
                break  # "growing": others use the card; the holder keeps taking memory as it frees up
            if p.poll() is not None:
                sys.exit(f"[gpu_hold] holder exited early (code {p.returncode}), see {d / 'holder.log'}")
            time.sleep(0.3)
        st = read_state(d)
        print(f"[gpu_hold] GPU {physical_gpu()}: holder pid {p.pid} {st.get('state', 'starting')}, "
              f"{st.get('held_gb', 0):.0f}/{st.get('target_gb', 0):.0f} GB ({a.fraction:.0%} of the card - "
              f"{a.margin_gb:.0f} GB margin). Stop with: bash scripts/gpu_hold.sh stop")
    elif a.cmd == "status":
        print(json.dumps(_status(d), indent=2))
    elif a.cmd == "stop":
        if pid is None:
            print(f"[gpu_hold] not running on GPU {physical_gpu()}")
            return
        os.kill(pid, signal.SIGTERM)
        for _ in range(100):
            if holder_pid(d) is None:
                break
            time.sleep(0.1)
        print(f"[gpu_hold] stopped (pid {pid}); GPU {physical_gpu()} share released")
    elif a.cmd == "run":
        fake = os.environ.get("MONET_GPU_HOLD_FAKE_GB")
        card = _FakeCard(float(fake)) if fake else None
        run_holder(a.fraction, a.margin_gb, a.poll, d, *((card.alloc, card.release, card.mem_info) if card else ()))


if __name__ == "__main__":
    main()
