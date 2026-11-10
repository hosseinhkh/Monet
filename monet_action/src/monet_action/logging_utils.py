"""Run logging/monitoring: text logs, status heartbeat, event stream and system metrics.

Every training or evaluation run directory gets (file contract shared with the dashboard,
see docs/DASHBOARD.md):

  logs/<kind>.log    human-readable log (python logging; warnings and tracebacks included)
  status.json        {"kind","name","variant","state","pid","host","started_at","updated_at",
                      "step","max_steps",...,"error"} rewritten atomically on every update
  events.jsonl       {"time","event",...} start / resume / checkpoint / val / sim_eval / finished / failed
  system.jsonl       {"time","gpus":[...],"cpu_percent","ram_used_gb",...} sampled every N seconds

Usage::

    with RunMonitor(run_dir, kind="train", name=cfg.name, variant=cfg.variant, max_steps=N) as mon:
        log = mon.logger
        ...
        mon.update(step=step, last_loss=loss)
        mon.event("checkpoint", path=str(ckpt))

Leaving the ``with`` block normally marks the run ``finished``; an exception marks it ``failed``
(traceback stored in status.json and the log) and KeyboardInterrupt marks it ``stopped``.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER_NAME = "monet_action"
_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(name)
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_monet_console", False)
               for h in logging.getLogger(LOGGER_NAME).handlers):
        h = logging.StreamHandler()
        h._monet_console = True  # type: ignore[attr-defined]
        h.setFormatter(logging.Formatter(_FMT, "%H:%M:%S"))
        root = logging.getLogger(LOGGER_NAME)
        root.addHandler(h)
        root.setLevel(logging.INFO)
        root.propagate = False
    return logger


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def _append_jsonl(path: Path, obj: Dict[str, Any], lock) -> None:
    line = json.dumps(obj, default=str)
    with lock:
        with open(path, "a") as f:
            f.write(line + "\n")


# ------------------------------------------------------------------------------ system stats
def _nvidia_smi() -> List[Dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    q = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []

    def num(x):
        try:
            return float(x)
        except ValueError:
            return None

    gpus = []
    for line in out.strip().splitlines():
        p = [s.strip() for s in line.split(",")]
        if len(p) < 7:
            continue
        mu, mt = num(p[3]), num(p[4])
        gpus.append({"index": int(num(p[0]) or 0), "name": p[1], "util_percent": num(p[2]),
                     "mem_used_gb": mu / 1024 if mu is not None else None,
                     "mem_total_gb": mt / 1024 if mt is not None else None,
                     "temp_c": num(p[5]), "power_w": num(p[6])})
    return gpus


class _CpuSampler:
    """CPU utilisation from /proc/stat deltas (psutil is used when installed)."""

    def __init__(self):
        self.prev = self._read()

    @staticmethod
    def _read():
        try:
            with open("/proc/stat") as f:
                vals = [float(v) for v in f.readline().split()[1:]]
            return sum(vals), vals[3] + (vals[4] if len(vals) > 4 else 0.0)
        except Exception:
            return None

    def percent(self) -> Optional[float]:
        try:
            import psutil  # type: ignore

            return float(psutil.cpu_percent(interval=None))
        except Exception:
            pass
        cur = self._read()
        if cur is None or self.prev is None:
            return None
        dt, di = cur[0] - self.prev[0], cur[1] - self.prev[1]
        self.prev = cur
        return None if dt <= 0 else 100.0 * (1.0 - di / dt)


def _meminfo() -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"ram_used_gb": None, "ram_total_gb": None, "proc_rss_gb": None}
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = float(v.split()[0]) / 1024 ** 2  # kB -> GB
        out["ram_total_gb"] = info.get("MemTotal")
        if "MemAvailable" in info and "MemTotal" in info:
            out["ram_used_gb"] = info["MemTotal"] - info["MemAvailable"]
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["proc_rss_gb"] = float(line.split()[1]) / 1024 ** 2
    except Exception:
        pass
    return out


def system_snapshot(path_for_disk: str = ".", cpu: Optional[_CpuSampler] = None) -> Dict[str, Any]:
    snap: Dict[str, Any] = {"time": time.time(), "gpus": _nvidia_smi()}
    snap["cpu_percent"] = cpu.percent() if cpu is not None else None
    snap.update(_meminfo())
    try:
        snap["disk_free_gb"] = shutil.disk_usage(path_for_disk).free / 1024 ** 3
    except Exception:
        snap["disk_free_gb"] = None
    try:
        snap["load_avg_1m"] = os.getloadavg()[0]
    except Exception:
        snap["load_avg_1m"] = None
    snap["torch_mem_allocated_gb"] = snap["torch_max_mem_allocated_gb"] = None
    try:
        import torch

        if torch.cuda.is_available():
            snap["torch_mem_allocated_gb"] = torch.cuda.memory_allocated() / 1024 ** 3
            snap["torch_max_mem_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    except Exception:
        pass
    return snap


# ------------------------------------------------------------------------------ monitor
class RunMonitor:
    def __init__(self, run_dir, kind: str = "train", name: str = "", variant: str = "",
                 max_steps: Optional[int] = None, system_interval_s: float = 30.0, extra: Optional[Dict] = None):
        self.dir = Path(run_dir)
        self.kind = kind
        self.logger = get_logger(f"{LOGGER_NAME}.{kind}")
        self.system_interval_s = system_interval_s
        self._lock = threading.RLock()  # shared by the main thread and the system-monitor thread
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._file_handler: Optional[logging.Handler] = None
        self.status: Dict[str, Any] = {
            "kind": kind, "name": name, "variant": variant, "state": "starting", "pid": os.getpid(),
            "host": socket.gethostname(), "started_at": time.time(), "updated_at": time.time(),
            "step": 0, "max_steps": max_steps, "error": None, "output_dir": str(self.dir),
        }
        if extra:
            self.status.update(extra)

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> "RunMonitor":
        (self.dir / "logs").mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(self.dir / "logs" / f"{self.kind}.log")
        fh.setFormatter(logging.Formatter(_FMT))
        logging.getLogger(LOGGER_NAME).addHandler(fh)
        logging.getLogger("py.warnings").addHandler(fh)
        logging.captureWarnings(True)
        self._file_handler = fh
        prev = self._read_previous_status()
        if prev is not None and prev.get("started_at"):
            self.status["first_started_at"] = prev.get("first_started_at", prev["started_at"])
        self.status["state"] = "running"
        self._write_status()
        self.event("start", pid=os.getpid(), host=self.status["host"])
        if self.system_interval_s and self.system_interval_s > 0:
            self._thread = threading.Thread(target=self._system_loop, name="system-monitor", daemon=True)
            self._thread.start()
        return self

    def finish(self, state: str = "finished", error: Optional[str] = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        with self._lock:
            self.status.update(state=state, error=error, finished_at=time.time())
            self._write_status()
        self.event(state, **({"error": error.splitlines()[-1] if error else None} if error else {}))
        if error:
            self.logger.error("run %s:\n%s", state, error)
        else:
            self.logger.info("run %s", state)
        if self._file_handler is not None:
            self._file_handler.flush()
            logging.getLogger(LOGGER_NAME).removeHandler(self._file_handler)
            logging.getLogger("py.warnings").removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None

    def __enter__(self) -> "RunMonitor":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.finish("finished")
        elif issubclass(exc_type, KeyboardInterrupt):
            self.finish("stopped", "KeyboardInterrupt")
        else:
            self.finish("failed", "".join(traceback.format_exception(exc_type, exc, tb)))
        return False  # never swallow exceptions

    # ---------------------------------------------------------------- updates
    def update(self, **fields: Any) -> None:
        with self._lock:
            self.status.update(fields)
            self._write_status()

    def event(self, name: str, **data: Any) -> None:
        rec = {"time": time.time(), "event": name, **data}
        _append_jsonl(self.dir / "events.jsonl", rec, self._lock)

    def _write_status(self) -> None:
        with self._lock:
            self.status["updated_at"] = time.time()
            self.dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(self.dir / "status.json", dict(self.status))

    def _read_previous_status(self) -> Optional[Dict]:
        p = self.dir / "status.json"
        try:
            return json.loads(p.read_text()) if p.exists() else None
        except Exception:
            return None

    def _system_loop(self) -> None:
        cpu = _CpuSampler()
        while not self._stop.is_set():
            try:
                snap = system_snapshot(str(self.dir), cpu)
                _append_jsonl(self.dir / "system.jsonl", snap, self._lock)
                # heartbeat: keeps updated_at fresh even during long validations / sim evals
                self._write_status()
            except Exception as e:  # monitoring must never kill a run
                self.logger.debug("system monitor error: %s", e)
            self._stop.wait(self.system_interval_s)
