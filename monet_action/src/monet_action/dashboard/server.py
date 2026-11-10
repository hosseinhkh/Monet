"""Monet-Action monitoring dashboard: a dependency-free (stdlib) HTTP server + static single-page UI.

  python -m monet_action.dashboard --runs runs --results results --port 8765 [--token SECRET]

It only *reads* the files that training / evaluation write (see docs/DASHBOARD.md for the file
contract), so it can run next to a live job, on a laptop over an SSH tunnel, or after the fact on a
downloaded debug bundle. Nothing here imports torch.

Security model: binds 127.0.0.1 by default; ``--token`` (or $MONET_DASHBOARD_TOKEN) is required on
every request (``?token=``, ``X-Dashboard-Token`` / ``Authorization: Bearer`` header, or the session
cookie set after the first authenticated request); files are served only from the two configured
roots, after resolving symlinks, and never as HTML.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import http.cookies
import http.server
import json
import math
import os
import re
import socket
import sys
import threading
import time
import traceback
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..bundle import write_bundle

STATIC_DIR = Path(__file__).resolve().parent / "static"
VERSION = "0.1.0"
STALE_AFTER_S = 600.0
MAX_POINTS = 2000
DISCOVERY_TTL_S = 2.0
RUN_MARKERS = ("config.json", "status.json")
EVAL_MARKERS = ("summary.json", "episodes.jsonl")
PRUNE_DIRS = {"logs", "videos", "traces", "__pycache__"}
SERIES_FILES = {
    "train_log": "train_log.jsonl",
    "val_log": "val_log.jsonl",
    "overfit_log": "overfit_log.jsonl",
    "sim_eval_log": "sim_eval_log.jsonl",
    "system": "system.jsonl",
    "events": "events.jsonl",
}
LOG_EXTS = {".log", ".txt", ".out", ".err"}
COOKIE_NAME = "monet_dash_session"
CONTENT_TYPES = {
    ".json": "application/json; charset=utf-8",
    ".jsonl": "text/plain; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
    ".out": "text/plain; charset=utf-8",
    ".err": "text/plain; charset=utf-8",
    ".yaml": "text/plain; charset=utf-8",
    ".yml": "text/plain; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".py": "text/plain; charset=utf-8",
    ".html": "text/plain; charset=utf-8",  # never render HTML found under the data roots
    ".htm": "text/plain; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".zip": "application/zip",
}
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
PAGE_CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self'; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
FILE_CSP = "sandbox; default-src 'none'; img-src 'self' data:; media-src 'self'; style-src 'unsafe-inline'"


# ======================================================================================
# helpers: JSON sanitising, safe paths, file reading
# ======================================================================================
def sanitize(o: Any) -> Any:
    """Make Python-json output (which may contain NaN / Infinity) valid strict JSON."""
    if isinstance(o, float):
        if math.isnan(o):
            return None
        if math.isinf(o):
            return "Infinity" if o > 0 else "-Infinity"
        return o
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    return o


def read_json(path: Path) -> Optional[Any]:
    try:
        with open(path, "rb") as f:
            return json.loads(f.read().decode("utf-8", "replace"))
    except (OSError, ValueError):
        return None


def safe_join(base: Path, rel: str) -> Optional[Path]:
    """Resolve ``rel`` under ``base``; None if it escapes ``base`` (``..``, absolute, symlinks, NUL)."""
    if rel is None or "\x00" in rel:
        return None
    parts = [p for p in rel.replace("\\", "/").split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return None
    try:
        base_r = base.resolve()
        target = base_r.joinpath(*parts).resolve() if parts else base_r
    except (OSError, RuntimeError, ValueError):
        return None
    if target != base_r and base_r not in target.parents:
        return None
    return target


def file_mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def tail_bytes(path: Path, max_bytes: int) -> Tuple[bytes, bool]:
    """Last ``max_bytes`` of a file and whether the head was cut."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        start = max(0, size - max_bytes)
        f.seek(start)
        return f.read(), start > 0


def last_jsonl_row(path: Path, predicate: Callable[[Dict], bool] = lambda r: True) -> Optional[Dict]:
    """Last complete JSONL row satisfying ``predicate`` (reads only the tail of the file)."""
    if not path.is_file():
        return None
    for nbytes in (64 * 1024, 1024 * 1024, 16 * 1024 * 1024):
        try:
            data, cut = tail_bytes(path, nbytes)
        except OSError:
            return None
        lines = data.split(b"\n")
        if cut:
            lines = lines[1:]  # first line may be partial
        for raw in reversed(lines):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if isinstance(row, dict) and predicate(row):
                return row
        if not cut:
            return None
    return None


def count_lines(path: Path) -> int:
    n = 0
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                n += chunk.count(b"\n")
    except OSError:
        return 0
    return n


def downsample(rows: List[Any], max_points: int = MAX_POINTS) -> Tuple[List[Any], bool]:
    """Uniform stride downsampling that always keeps the first and the latest row (<= max_points)."""
    n = len(rows)
    if n <= max_points:
        return rows, False
    stride = math.ceil(n / (max_points - 1))
    out = rows[::stride]
    if out[-1] is not rows[-1]:
        out.append(rows[-1])
    return out, True


class JsonlIndex:
    """Per-file line-offset cache: ``read(path, since)`` costs O(new bytes), not O(file).

    Only newline-terminated lines are indexed, so a line that is still being written is never
    returned (it is picked up by the next poll once complete). Malformed complete lines are
    skipped but still counted, so the client's ``since`` cursor always advances.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}

    def read(self, path: Path, since: int = 0) -> Dict[str, Any]:
        key = str(path)
        try:
            st = path.stat()
        except OSError:
            return {"exists": False, "rows": [], "next": 0, "total": 0, "file_id": None, "reset": since > 0}
        file_id = f"{st.st_dev}:{st.st_ino}"
        with self._lock:
            ent = self._cache.get(key)
            if ent is None or ent["file_id"] != file_id or st.st_size < ent["end"]:
                ent = {"file_id": file_id, "offsets": [], "end": 0}
            if st.st_size > ent["end"]:
                with open(path, "rb") as f:
                    f.seek(ent["end"])
                    chunk = f.read(st.st_size - ent["end"])
                pos, start = ent["end"], 0
                while True:
                    nl = chunk.find(b"\n", start)
                    if nl < 0:
                        break
                    ent["offsets"].append(pos + start)
                    start = nl + 1
                ent["end"] = pos + start
            self._cache[key] = ent
            total = len(ent["offsets"])
            reset = since < 0 or since > total
            if reset:
                since = 0
            first = ent["offsets"][since] if since < total else ent["end"]
            end = ent["end"]
        rows: List[Dict] = []
        if end > first:
            with open(path, "rb") as f:
                f.seek(first)
                data = f.read(end - first)
            for raw in data.split(b"\n"):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        return {"exists": True, "rows": rows, "next": total, "total": total, "file_id": file_id, "reset": reset}


# ======================================================================================
# discovery and summaries
# ======================================================================================
def _walk_dirs(root: Path, max_depth: int):
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts)
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and not d.startswith("ckpt_") and d not in PRUNE_DIRS)
        if depth >= max_depth:
            dirnames[:] = []
        yield rel, Path(dirpath), dirnames, set(filenames)


def discover_runs(root: Path, max_depth: int = 3) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for rel, path, dirnames, files in _walk_dirs(root, max_depth):
        if rel.parts and any(m in files for m in RUN_MARKERS):
            out[rel.as_posix()] = path
            dirnames[:] = []  # subdirectories of a run are not separate runs
    return out


def discover_evals(root: Path, max_depth: int = 4) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    eval_dir = root / "eval"
    if eval_dir.is_dir():  # every subdir of results/eval, even before summary.json exists
        for d in sorted(eval_dir.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                out[d.relative_to(root).as_posix()] = d
    for rel, path, dirnames, files in _walk_dirs(root, max_depth):
        if rel.parts and any(m in files for m in EVAL_MARKERS):
            out.setdefault(rel.as_posix(), path)
            dirnames[:] = []
    return dict(sorted(out.items()))


def _num(x) -> Optional[float]:
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def key_config(cfg: Dict) -> List[Dict[str, Any]]:
    """Grouped, human-sized view of the settings that change the meaning of a run."""
    if not isinstance(cfg, dict):
        return []
    m, c, a = cfg.get("monet", {}) or {}, cfg.get("conditioner", {}) or {}, cfg.get("action_head", {}) or {}
    t, d, p, s = cfg.get("train", {}) or {}, cfg.get("data", {}) or {}, cfg.get("prompt", {}) or {}, cfg.get("state", {}) or {}
    lora = (f"r={m.get('lora_r')} layers {m.get('lora_layers')} {m.get('lora_target_modules')}"
            if m.get("lora_enabled") else "off")
    groups = [
        ("Run", [("name", cfg.get("name")), ("variant", cfg.get("variant")), ("description", cfg.get("description"))]),
        ("Monet", [("model", m.get("model_path")), ("K latents", m.get("num_latents")),
                   ("reasoning", m.get("reasoning_mode")), ("convention", m.get("latent_convention")),
                   ("recurrence", m.get("recurrence_impl")), ("feedback grad", m.get("feedback_grad")),
                   ("LoRA", lora), ("grad checkpointing", m.get("gradient_checkpointing")), ("dtype", m.get("dtype"))]),
        ("Conditioning", [("feature source", c.get("feature_source")), ("cond dim", c.get("cond_dim")),
                          ("state into Monet", s.get("inject_into_monet")), ("state to head", s.get("state_to_action_head")),
                          ("state tokens", p.get("num_state_tokens")), ("image size", p.get("image_size"))]),
        ("Action head", [("type", a.get("type")), ("horizon x dim", f"{a.get('action_horizon')} x {a.get('action_dim')}"),
                         ("width / layers / heads", f"{a.get('hidden_size')} / {a.get('num_layers')} / {a.get('num_heads')}"),
                         ("Euler steps", a.get("num_inference_steps"))]),
        ("Optimisation", [("max steps", t.get("max_steps")),
                          ("batch x accum", f"{t.get('batch_size')} x {t.get('grad_accum')}"),
                          ("lr head / state / lora", f"{t.get('lr_head')} / {t.get('lr_state')} / {t.get('lr_lora')}"),
                          ("warmup", t.get("warmup_steps")), ("precision", t.get("mixed_precision")),
                          ("seed", t.get("seed")), ("init from", t.get("init_from"))]),
        ("Data", [("root", d.get("root")), ("val fraction", d.get("val_fraction")), ("split seed", d.get("split_seed")),
                  ("file limit", d.get("file_limit")), ("max windows", d.get("max_windows"))]),
    ]
    return [{"group": g, "items": [{"key": k, "value": v} for k, v in items]} for g, items in groups]


def _ckpt_step(name: str) -> Optional[int]:
    m = re.fullmatch(r"ckpt_(\d+)", name)
    return int(m.group(1)) if m else None


def list_checkpoints(path: Path) -> List[Dict[str, Any]]:
    last = None
    try:
        last = (path / "last").read_text().strip() or None
    except OSError:
        pass
    out = []
    for d in sorted(path.glob("ckpt_*")):
        step = _ckpt_step(d.name)
        if step is None or not d.is_dir():
            continue
        files = []
        for f in sorted(d.iterdir()):
            if f.is_file():
                files.append({"name": f.name, "size": f.stat().st_size})
        meta = read_json(d / "meta.json") or {}
        out.append({"name": d.name, "step": step, "is_last": d.name == last, "files": files,
                    "size": sum(f["size"] for f in files), "consumed_samples": meta.get("consumed_samples"),
                    "mtime": file_mtime(d)})
    return out


def list_files(path: Path, limit: int = 3000) -> Tuple[List[Dict[str, Any]], bool]:
    out: List[Dict[str, Any]] = []
    truncated = False
    root_r = path.resolve()
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__pycache__")
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            p = Path(dirpath) / fn
            try:
                real = p.resolve()
                if real != root_r and root_r not in real.parents:
                    continue
                st = real.stat()
            except OSError:
                continue
            out.append({"path": p.relative_to(path).as_posix(), "size": st.st_size, "mtime": st.st_mtime})
            if len(out) >= limit:
                return out, True
    return out, truncated


def list_logs(path: Path) -> List[Dict[str, Any]]:
    """Text logs of a run/eval: ``logs/**`` first, then top-level *.log / *.txt / *.out / *.err."""
    cands = list(path.glob("*"))
    if (path / "logs").is_dir():
        cands += list((path / "logs").rglob("*"))
    logs = []
    for f in cands:
        if f.is_file() and f.suffix.lower() in LOG_EXTS and not f.name.startswith("."):
            logs.append({"path": f.relative_to(path).as_posix(), "size": f.stat().st_size, "mtime": file_mtime(f)})
    logs.sort(key=lambda x: (not x["path"].startswith("logs/"), x["path"]))
    return logs


def log_tail(path: Path, tail: int = 500, q: Optional[str] = None, max_bytes: int = 8 * 1024 * 1024) -> Dict:
    tail = max(1, min(int(tail), 20000))
    window = 256 * 1024
    data, cut = b"", False
    while True:  # grow the window until it holds ``tail`` lines (or the cap / file start is reached)
        data, cut = tail_bytes(path, window)
        if not cut or data.count(b"\n") > tail or window >= max_bytes or q:
            if q and window < max_bytes and cut:
                window = max_bytes
                continue
            break
        window *= 4
    text = data.decode("utf-8", "replace")
    lines = text.splitlines()
    if cut and lines:
        lines = lines[1:]
    n_scanned = len(lines)
    if q:
        ql = q.lower()
        lines = [ln for ln in lines if ql in ln.lower()]
    st = path.stat()
    return {"path": path.name, "lines": lines[-tail:], "n_matched": len(lines) if q else None,
            "n_scanned": n_scanned, "truncated_head": cut, "size": st.st_size, "mtime": st.st_mtime, "query": q}


def run_summary(name: str, path: Path, now: float) -> Dict[str, Any]:
    status = read_json(path / "status.json") or {}
    cfg = read_json(path / "config.json") or {}
    m, t, a, c = (cfg.get("monet") or {}), (cfg.get("train") or {}), (cfg.get("action_head") or {}), (cfg.get("conditioner") or {})
    last = last_jsonl_row(path / "train_log.jsonl", lambda r: "loss" in r and "event" not in r)
    mtimes = [file_mtime(path / f) for f in ("status.json", "train_log.jsonl", "events.jsonl", "val_log.jsonl",
                                             "overfit_log.jsonl", "system.jsonl", "config.json")]
    mtimes = [x for x in mtimes if x]
    updated_at = _num(status.get("updated_at")) or (max(mtimes) if mtimes else None)
    max_steps = status.get("max_steps") if status.get("max_steps") is not None else t.get("max_steps")
    step = status.get("step") if status.get("step") is not None else (last or {}).get("step")
    state = status.get("state")
    inferred = False
    if not state:
        inferred = True
        if max_steps and step is not None and step >= max_steps:
            state = "finished"
        elif updated_at and now - updated_at < STALE_AFTER_S and last:
            state = "running"
        else:
            state = "unknown"
    stale = bool(state == "running" and updated_at and now - updated_at > STALE_AFTER_S)
    last_loss = status.get("last_loss") if status.get("last_loss") is not None else (last or {}).get("loss")
    progress = (step / max_steps) if (isinstance(step, (int, float)) and max_steps) else None
    return {
        "name": name, "kind": "train", "variant": status.get("variant") or cfg.get("variant"),
        "description": cfg.get("description"), "config_name": cfg.get("name"),
        "state": state, "display_state": "stale" if stale else state, "stale": stale, "state_inferred": inferred,
        "step": step, "max_steps": max_steps, "progress": progress, "samples": status.get("samples", (last or {}).get("samples")),
        "eta_s": status.get("eta_s"), "last_loss": last_loss, "updated_at": updated_at,
        "updated_ago_s": (now - updated_at) if updated_at else None, "started_at": status.get("started_at"),
        "pid": status.get("pid"), "host": status.get("host"), "error": status.get("error"),
        "output_dir": status.get("output_dir"),
        "K": m.get("num_latents"), "lora": bool(m.get("lora_enabled")), "reasoning_mode": m.get("reasoning_mode"),
        "latent_convention": m.get("latent_convention"), "feature_source": c.get("feature_source"),
        "head": a.get("type"), "model_path": m.get("model_path"),
        "has": {k: (path / f).is_file() for k, f in SERIES_FILES.items()},
    }


def eval_summary(name: str, path: Path, now: float, run_names: Optional[Dict[str, Path]] = None) -> Dict[str, Any]:
    s = read_json(path / "summary.json") or {}
    st = read_json(path / "status.json") or {}
    ec = read_json(path / "eval_config.json") or {}
    model = ec.get("model_config") or {}
    mm = model.get("monet") or {}
    ev = ec.get("eval") or {}
    n_done = st.get("episodes_done")
    if n_done is None:
        n_done = count_lines(path / "episodes.jsonl") if (path / "episodes.jsonl").is_file() else 0
    n_total = st.get("episodes_total")
    if n_total is None and ev.get("episodes_per_task") and isinstance(ev.get("seeds"), list):
        n_tasks = len(ev["task_ids"]) if isinstance(ev.get("task_ids"), list) else 10
        n_total = ev["episodes_per_task"] * len(ev["seeds"]) * n_tasks
    state = st.get("state") or ("finished" if s else ("running" if n_done else "unknown"))
    mtimes = [x for x in (file_mtime(path / f) for f in ("status.json", "episodes.jsonl", "summary.json")) if x]
    updated_at = _num(st.get("updated_at")) or (max(mtimes) if mtimes else None)
    stale = bool(state == "running" and updated_at and now - updated_at > STALE_AFTER_S)
    run_dir = (model.get("train") or {}).get("output_dir")
    run_name = None
    if run_names:
        base = Path(run_dir).name if run_dir else None
        for cand in (base, model.get("name")):
            if cand and cand in run_names:
                run_name = cand
                break
    lat = s.get("latency_ms_per_call") or {}
    return {
        "name": name, "kind": "eval", "state": state, "display_state": "stale" if stale else state, "stale": stale,
        "episodes_done": n_done, "episodes_total": n_total, "updated_at": updated_at,
        "updated_ago_s": (now - updated_at) if updated_at else None,
        "success_rate": s.get("success_rate"), "success_ci95": s.get("success_ci95"), "n_episodes": s.get("n_episodes"),
        "n_success": sum(int(v.get("successes", 0)) for v in (s.get("per_task") or {}).values()) if s else None,
        "gpu_seconds_per_success": s.get("gpu_seconds_per_success"),
        "gpu_seconds_per_episode": s.get("gpu_seconds_per_episode"),
        "latency": {k: {"p50": (v or {}).get("p50"), "p95": (v or {}).get("p95")} for k, v in lat.items()},
        "intervention": s.get("intervention", ev.get("intervention")), "tag": s.get("tag"),
        "num_inference_steps": s.get("num_inference_steps", ev.get("num_inference_steps")),
        "execution_prefix": s.get("execution_prefix", ev.get("execution_prefix")),
        "model": {"name": model.get("name"), "variant": model.get("variant"), "run_dir": run_dir,
                  "K": mm.get("num_latents"), "lora": mm.get("lora_enabled")},
        "run_name": run_name, "has_videos": (path / "videos").is_dir(),
        "has": {"summary": bool(s), "offline": (path / "offline.json").is_file(), "latency": (path / "latency.json").is_file()},
    }


def _episode_row(r: Dict, videos: Dict[str, str], traces: Dict[str, str]) -> Dict[str, Any]:
    lat = r.get("call_latency_ms") or []
    totals = [sum(v for v in d.values() if isinstance(v, (int, float))) for d in lat if isinstance(d, dict)]
    totals_sorted = sorted(totals)
    key = f"seed{r.get('seed')}_task{r.get('task_id')}_init{r.get('init_state_idx')}"
    bs = [b for b in (r.get("batch_sizes") or []) if isinstance(b, (int, float))]
    return {
        "key": key, "task_id": r.get("task_id"), "init_state_idx": r.get("init_state_idx"), "seed": r.get("seed"),
        "success": bool(r.get("success")), "env_steps": r.get("env_steps"), "action_calls": r.get("action_calls"),
        "sim_time_ms": r.get("sim_time_ms"), "wall_s": r.get("wall_s"),
        "failure_tags": r.get("failure_tags") or [],
        "call_ms_mean": (sum(totals) / len(totals)) if totals else None,
        "call_ms_p95": totals_sorted[min(len(totals_sorted) - 1, int(0.95 * len(totals_sorted)))] if totals else None,
        "batch_size_mean": (sum(bs) / len(bs)) if bs else None,
        "video": videos.get(key), "trace": traces.get(key),
    }


def eval_detail(name: str, path: Path, now: float, run_names: Dict[str, Path]) -> Dict[str, Any]:
    videos, traces = {}, {}
    for v in sorted((path / "videos").glob("*")) if (path / "videos").is_dir() else []:
        m = re.match(r"(seed-?\d+_task-?\d+_init-?\d+)", v.name)
        if m and v.is_file():
            videos.setdefault(m.group(1), v.relative_to(path).as_posix())
    for t in sorted((path / "traces").glob("*.jsonl")) if (path / "traces").is_dir() else []:
        traces[t.stem] = t.relative_to(path).as_posix()
    episodes = []
    ep_file = path / "episodes.jsonl"
    if ep_file.is_file():
        for raw in ep_file.read_bytes().split(b"\n"):
            raw = raw.strip()
            if not raw:
                continue
            try:
                r = json.loads(raw)
            except ValueError:
                continue
            if isinstance(r, dict):
                episodes.append(_episode_row(r, videos, traces))
    files, trunc = list_files(path)
    return {
        "summary_row": eval_summary(name, path, now, run_names),
        "summary": read_json(path / "summary.json"), "status": read_json(path / "status.json"),
        "eval_config": read_json(path / "eval_config.json"), "offline": read_json(path / "offline.json"),
        "latency": read_json(path / "latency.json"), "episodes": episodes,
        "videos": sorted(v.relative_to(path).as_posix() for v in (path / "videos").glob("*")
                         if v.is_file() and v.suffix.lower() in (".mp4", ".webm")) if (path / "videos").is_dir() else [],
        "logs": list_logs(path), "files": files, "files_truncated": trunc,
    }


def run_detail(name: str, path: Path, now: float) -> Dict[str, Any]:
    cfg = read_json(path / "config.json")
    split = read_json(path / "split.json")
    env = read_json(path / "environment.json")
    files, trunc = list_files(path)
    return {
        "summary": run_summary(name, path, now),
        "status": read_json(path / "status.json"), "config": cfg, "key_config": key_config(cfg or {}),
        "param_report": read_json(path / "param_report.json"), "gradient_audit": read_json(path / "gradient_audit.json"),
        "memory_profile": read_json(path / "memory_profile.json"), "representation": read_json(path / "representation.json"),
        "environment": env,
        "split": {k: (len(v) if isinstance(v, list) else v) for k, v in split.items()} if isinstance(split, dict) and split else None,
        "checkpoints": list_checkpoints(path), "logs": list_logs(path),
        "series": {k: {"exists": (path / f).is_file(), "size": (path / f).stat().st_size if (path / f).is_file() else 0}
                   for k, f in SERIES_FILES.items()},
        "files": files, "files_truncated": trunc,
    }


COMPARE_TRAIN_KEYS = ("step", "loss", "fm_loss", "mse_loss", "grad_norm", "lr_scale", "time")


def audit_listing(results_root: Path) -> Dict[str, Any]:
    out: Dict[str, List[Dict[str, Any]]] = {"audit": [], "reports": []}
    audit = results_root / "audit"
    if audit.is_dir():
        for f in sorted(audit.rglob("*")):
            if f.is_file() and not any(p.startswith(".") for p in f.relative_to(results_root).parts):
                out["audit"].append({"path": f.relative_to(results_root).as_posix(), "size": f.stat().st_size,
                                     "mtime": file_mtime(f), "ext": f.suffix.lower()})
    if results_root.is_dir():
        for f in sorted(results_root.glob("*")):
            if f.is_file() and f.suffix.lower() in (".md", ".json", ".txt"):
                out["reports"].append({"path": f.relative_to(results_root).as_posix(), "size": f.stat().st_size,
                                       "mtime": file_mtime(f), "ext": f.suffix.lower()})
    return out


# ======================================================================================
# HTTP layer
# ======================================================================================
class _ChunkedWriter:
    """File-like object writing HTTP/1.1 chunked transfer encoding (zipfile treats it as non-seekable)."""

    def __init__(self, wfile, buffer_size: int = 256 * 1024):
        self.wfile, self.buf, self.buffer_size = wfile, bytearray(), buffer_size

    def write(self, data) -> int:
        self.buf += data
        if len(self.buf) >= self.buffer_size:
            self._emit()
        return len(data)

    def _emit(self) -> None:
        if self.buf:
            self.wfile.write(f"{len(self.buf):X}\r\n".encode("ascii") + bytes(self.buf) + b"\r\n")
            self.buf = bytearray()

    def flush(self) -> None:
        self._emit()
        self.wfile.flush()

    def close(self) -> None:
        self._emit()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


class HttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


class DashboardServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, runs_root: Path, results_root: Path, token: Optional[str] = None,
                 refresh: int = 10, quiet: bool = False, verbose: bool = False):
        self.runs_root = Path(runs_root)
        self.results_root = Path(results_root)
        self.token = token or None
        self.session = hmac.new(token.encode(), b"monet-action-dashboard", hashlib.sha256).hexdigest() if token else None
        self.refresh = int(refresh)
        self.quiet = quiet
        self.verbose = verbose and not quiet
        self.jsonl = JsonlIndex()
        self._disc_lock = threading.Lock()
        self._disc: Dict[str, Tuple[float, Dict[str, Path]]] = {}
        super().__init__(addr, DashboardHandler)

    @property
    def roots(self) -> Dict[str, Path]:
        return {"runs": self.runs_root, "results": self.results_root}

    def runs(self) -> Dict[str, Path]:
        return self._cached("runs", lambda: discover_runs(self.runs_root))

    def evals(self) -> Dict[str, Path]:
        return self._cached("evals", lambda: discover_evals(self.results_root))

    def _cached(self, key: str, fn) -> Dict[str, Path]:
        now = time.monotonic()
        with self._disc_lock:
            hit = self._disc.get(key)
            if hit and now - hit[0] < DISCOVERY_TTL_S:
                return hit[1]
        val = fn()
        with self._disc_lock:
            self._disc[key] = (now, val)
        return val


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    server: DashboardServer
    server_version = f"MonetActionDashboard/{VERSION}"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------------ plumbing
    def log_request(self, code="-", size="-"):
        # default: only failed requests (auto-refresh would otherwise flood the console)
        failed = isinstance(code, int) and code >= 400 or (isinstance(code, str) and code[:1] in "45")
        if self.server.verbose or failed:
            super().log_request(code, size)

    def log_message(self, fmt, *args):
        if not self.server.quiet:
            sys.stderr.write("[dashboard] %s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):
        self._dispatch(head=False)

    def do_HEAD(self):
        self._dispatch(head=True)

    def _dispatch(self, head: bool) -> None:
        self._head = head
        self._extra_headers: List[Tuple[str, str]] = []
        try:
            url = urllib.parse.urlsplit(self.path)
            query = urllib.parse.parse_qs(url.query, keep_blank_values=True)
            if not self._authorized(query):
                raise HttpError(401, "authentication required: open the dashboard with ?token=<token>")
            self._route(url.path, query)
        except HttpError as e:
            self._send_json({"error": e.message, "status": e.status}, e.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # pragma: no cover - surfaced to the client and the console
            traceback.print_exc()
            try:
                self._send_json({"error": f"{type(e).__name__}: {e}", "status": 500}, 500)
            except Exception:
                pass

    def _authorized(self, query: Dict[str, List[str]]) -> bool:
        token = self.server.token
        if not token:
            return True
        cands = list(query.get("token", []))
        cands.append(self.headers.get("X-Dashboard-Token", ""))
        auth = self.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            cands.append(auth[7:].strip())
        for c in cands:
            if c and hmac.compare_digest(c.encode(), token.encode()):
                self._extra_headers.append(
                    ("Set-Cookie", f"{COOKIE_NAME}={self.server.session}; Path=/; HttpOnly; SameSite=Strict"))
                return True
        try:
            cookie = http.cookies.SimpleCookie(self.headers.get("Cookie", ""))
        except http.cookies.CookieError:
            return False
        morsel = cookie.get(COOKIE_NAME)
        return bool(morsel and hmac.compare_digest(morsel.value.encode(), self.server.session.encode()))

    def _common_headers(self, ctype: str, length: Optional[int], extra: Optional[List[Tuple[str, str]]] = None) -> None:
        self.send_header("Content-Type", ctype)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        for k, v in (extra or []) + self._extra_headers:
            self.send_header(k, v)
        self.end_headers()

    def _send_bytes(self, body: bytes, ctype: str, status: int = 200, extra=None) -> None:
        self.send_response(status)
        self._common_headers(ctype, len(body), extra)
        if not self._head:
            self.wfile.write(body)

    def _send_json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(sanitize(obj), allow_nan=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status, [("Cache-Control", "no-store")])

    # ------------------------------------------------------------------ routing
    def _route(self, raw_path: str, q: Dict[str, List[str]]) -> None:
        if raw_path in ("/", "/index.html"):
            return self._static("index.html")
        if raw_path == "/favicon.ico":
            return self._send_bytes(b"", "image/x-icon", 204)
        if raw_path.startswith("/static/"):
            return self._static(urllib.parse.unquote(raw_path[len("/static/"):]))
        if raw_path.startswith("/files/"):
            return self._files(raw_path[len("/files/"):], q)
        if raw_path.startswith("/api/"):
            return self._api(raw_path[len("/api/"):], q)
        raise HttpError(404, "not found")

    def _static(self, rel: str) -> None:
        p = safe_join(STATIC_DIR, rel)
        if p is None or not p.is_file():
            raise HttpError(404, "not found")
        extra = [("Cache-Control", "no-cache")]
        if p.suffix == ".html":
            extra.append(("Content-Security-Policy", PAGE_CSP))
        self._send_bytes(p.read_bytes(), STATIC_TYPES.get(p.suffix, "application/octet-stream"), 200, extra)

    def _files(self, rest: str, q: Dict[str, List[str]]) -> None:
        root_name, _, rel = rest.partition("/")
        root = self.server.roots.get(urllib.parse.unquote(root_name))
        if root is None:
            raise HttpError(404, "unknown root")
        target = safe_join(root, urllib.parse.unquote(rel))
        if target is None:
            raise HttpError(403, "path outside the dashboard roots")
        if not target.is_file():
            raise HttpError(404, "file not found")
        size = target.stat().st_size
        ctype = CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
        extra = [("Content-Security-Policy", FILE_CSP), ("Accept-Ranges", "bytes"), ("Cache-Control", "no-cache")]
        if _flag(q, "download"):
            extra.append(("Content-Disposition", f'attachment; filename="{_ascii_name(target.name)}"'))
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng and size > 0:
            m = re.fullmatch(r"\s*bytes=(\d*)-(\d*)\s*", rng)
            if not m or (m.group(1) == "" and m.group(2) == ""):
                raise HttpError(416, "invalid range")
            if m.group(1) == "":  # suffix range: last N bytes
                start = max(0, size - int(m.group(2)))
            else:
                start = int(m.group(1))
                end = min(size - 1, int(m.group(2))) if m.group(2) else size - 1
            if start > end or start >= size:
                self.send_response(416)
                self._common_headers("text/plain", 0, [("Content-Range", f"bytes */{size}")])
                return
            status = 206
            extra.append(("Content-Range", f"bytes {start}-{end}/{size}"))
        length = max(0, end - start + 1) if size > 0 else 0
        self.send_response(status)
        self._common_headers(ctype, length, extra)
        if self._head or length == 0:
            return
        with open(target, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    # ------------------------------------------------------------------ API
    def _api(self, rest: str, q: Dict[str, List[str]]) -> None:
        segs = [urllib.parse.unquote(s) for s in rest.split("/") if s != ""]
        if not segs:
            raise HttpError(404, "not found")
        now = time.time()
        head, args = segs[0], segs[1:]
        srv = self.server
        if head == "config":
            return self._send_json({"version": VERSION, "refresh": srv.refresh, "now": now,
                                    "runs_root": str(srv.runs_root.resolve()), "results_root": str(srv.results_root.resolve()),
                                    "token_required": bool(srv.token), "stale_after_s": STALE_AFTER_S,
                                    "max_points": MAX_POINTS, "host": socket.gethostname()})
        if head == "runs":
            runs = srv.runs()
            return self._send_json({"now": now, "root": str(srv.runs_root),
                                    "runs": [run_summary(n, p, now) for n, p in runs.items()]})
        if head == "run":
            sub = None
            if args and args[-1] in ("series", "log", "files") and len(args) > 1:
                sub, args = args[-1], args[:-1]
            name = "/".join(args)
            path = srv.runs().get(name)
            if path is None:
                raise HttpError(404, f"unknown run '{name}'")
            if sub is None:
                return self._send_json({"now": now, "name": name, **run_detail(name, path, now)})
            if sub == "series":
                return self._send_json(self._series(path, q))
            if sub == "log":
                return self._send_json(self._log(path, q))
            files, trunc = list_files(path)
            return self._send_json({"files": files, "truncated": trunc})
        if head == "evals":
            runs = srv.runs()
            return self._send_json({"now": now, "root": str(srv.results_root),
                                    "evals": [eval_summary(n, p, now, runs) for n, p in srv.evals().items()]})
        if head == "eval":
            sub = None
            if args and args[-1] == "log" and len(args) > 1:
                sub, args = "log", args[:-1]
            name = "/".join(args)
            path = srv.evals().get(name)
            if path is None:
                raise HttpError(404, f"unknown evaluation '{name}'")
            if sub == "log":
                return self._send_json(self._log(path, q))
            return self._send_json({"now": now, "name": name, **eval_detail(name, path, now, srv.runs())})
        if head == "compare":
            return self._send_json(self._compare(q, now))
        if head == "audit":
            return self._send_json(audit_listing(srv.results_root))
        if head == "bundle":
            return self._bundle(args, q)
        raise HttpError(404, "unknown API endpoint")

    def _series(self, path: Path, q: Dict[str, List[str]]) -> Dict[str, Any]:
        key = _first(q, "file", "train_log")
        if key not in SERIES_FILES:
            raise HttpError(400, f"file must be one of {sorted(SERIES_FILES)}")
        try:
            since = int(_first(q, "since", "0"))
            max_points = max(10, min(int(_first(q, "max_points", str(MAX_POINTS))), MAX_POINTS))
        except ValueError:
            raise HttpError(400, "since / max_points must be integers") from None
        res = self.server.jsonl.read(path / SERIES_FILES[key], since)
        rows, ds = downsample(res["rows"], max_points)
        return {"file": key, "exists": res["exists"], "rows": rows, "next": res["next"], "total": res["total"],
                "file_id": res["file_id"], "reset": res["reset"], "downsampled": ds, "n_rows_raw": len(res["rows"])}

    def _log(self, path: Path, q: Dict[str, List[str]]) -> Dict[str, Any]:
        rel = _first(q, "file", None)
        logs = list_logs(path)
        if rel is None:
            if not logs:
                return {"path": None, "lines": [], "available": [], "message": "no log files"}
            rel = logs[0]["path"]
        target = safe_join(path, rel if "/" in rel or (path / rel).is_file() else f"logs/{rel}")
        if target is None:
            raise HttpError(403, "path outside the run directory")
        if not target.is_file() or target.suffix.lower() not in LOG_EXTS:
            raise HttpError(404, f"log '{rel}' not found")
        try:
            tail = int(_first(q, "tail", "500"))
        except ValueError:
            raise HttpError(400, "tail must be an integer") from None
        out = log_tail(target, tail, _first(q, "q", None) or None)
        out["path"] = target.relative_to(path.resolve()).as_posix()
        out["available"] = logs
        return out

    def _compare(self, q: Dict[str, List[str]], now: float) -> Dict[str, Any]:
        names = [n for n in _first(q, "runs", "").split(",") if n]
        runs = self.server.runs()
        out_runs = []
        for n in names[:12]:
            p = runs.get(n)
            if p is None:
                continue
            series = {}
            for key in ("train_log", "val_log", "overfit_log", "sim_eval_log"):
                res = self.server.jsonl.read(p / SERIES_FILES[key], 0)
                rows = res["rows"]
                if key == "train_log":
                    rows = [{k: r.get(k) for k in COMPARE_TRAIN_KEYS if k in r} for r in rows
                            if "loss" in r and "event" not in r]
                series[key], _ = downsample(rows, 1000)
            out_runs.append({"name": n, "summary": run_summary(n, p, now),
                             "key_config": key_config(read_json(p / "config.json") or {}), "series": series})
        evals = [eval_summary(n, p, now, runs) for n, p in self.server.evals().items()]
        linked = [e for e in evals if e["run_name"] in names]
        return {"now": now, "runs": out_runs, "evals": linked, "evals_linked_only": True,
                "missing": [n for n in names if n not in runs]}

    def _bundle(self, args: List[str], q: Dict[str, List[str]]) -> None:
        if not args:
            raise HttpError(404, "bundle kind required: run|eval|all")
        kind, name = args[0], "/".join(args[1:])
        ckpt, vids = _flag(q, "checkpoints"), _flag(q, "videos")
        if kind == "run":
            path = self.server.runs().get(name)
            if path is None:
                raise HttpError(404, f"unknown run '{name}'")
            sources = [(path, f"runs/{name}")]
        elif kind == "eval":
            path = self.server.evals().get(name)
            if path is None:
                raise HttpError(404, f"unknown evaluation '{name}'")
            sources, ckpt = [(path, f"results/{name}")], False
        elif kind == "all":
            sources, ckpt, vids = [(self.server.runs_root, "runs"), (self.server.results_root, "results")], False, False
            name = "all"
        else:
            raise HttpError(404, "bundle kind must be run|eval|all")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        fname = _ascii_name(f"monet_action_{kind}_{name.replace('/', '_')}{'_ckpt' if ckpt else ''}_{stamp}.zip")
        self.send_response(200)
        self.send_header("Transfer-Encoding", "chunked")
        self._common_headers("application/zip", None, [("Content-Disposition", f'attachment; filename="{fname}"'),
                                                       ("Cache-Control", "no-store")])
        if self._head:
            return
        writer = _ChunkedWriter(self.wfile)
        write_bundle(writer, sources, include_checkpoints=ckpt, include_videos=vids,
                     extra={"requested_via": "dashboard", "kind": kind, "name": name})
        writer.close()


def _first(q: Dict[str, List[str]], key: str, default):
    v = q.get(key)
    return v[0] if v else default


def _flag(q: Dict[str, List[str]], key: str) -> bool:
    return _first(q, key, "0").lower() in ("1", "true", "yes", "on")


def _ascii_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:180] or "download"


# ======================================================================================
# entry points
# ======================================================================================
def make_server(runs: str = "runs", results: str = "results", host: str = "127.0.0.1", port: int = 8765,
                token: Optional[str] = None, refresh: int = 10, quiet: bool = False,
                verbose: bool = False) -> DashboardServer:
    return DashboardServer((host, port), Path(runs), Path(results), token=token, refresh=refresh, quiet=quiet,
                           verbose=verbose)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Monet-Action training / evaluation monitoring dashboard")
    ap.add_argument("--runs", default="runs", help="root holding training run directories (default: runs)")
    ap.add_argument("--results", default="results", help="root holding evaluation results (default: results)")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; use an SSH tunnel)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token", default=os.environ.get("MONET_DASHBOARD_TOKEN"),
                    help="require this token on every request (default: $MONET_DASHBOARD_TOKEN)")
    ap.add_argument("--refresh", type=int, default=10, help="UI auto-refresh period in seconds")
    ap.add_argument("--quiet", action="store_true", help="no request logging at all")
    ap.add_argument("--verbose", action="store_true", help="log every request (default: failed requests only)")
    a = ap.parse_args(argv)
    if a.host not in ("127.0.0.1", "localhost", "::1") and not a.token:
        print(f"WARNING: binding {a.host} without --token exposes your logs to the network. "
              "Prefer the default 127.0.0.1 + `ssh -L`.", file=sys.stderr)
    try:
        srv = make_server(a.runs, a.results, a.host, a.port, a.token, a.refresh, a.quiet, a.verbose)
    except OSError as e:
        print(f"error: cannot bind {a.host}:{a.port} ({e}); try --port", file=sys.stderr)
        return 1
    host, port = srv.server_address[:2]
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}/"
    print(f"Monet-Action dashboard on {url}{'?token=' + a.token if a.token else ''}")
    print(f"  runs:    {Path(a.runs).resolve()}\n  results: {Path(a.results).resolve()}")
    if host in ("127.0.0.1", "localhost"):
        print(f"  remote:  ssh -L {port}:localhost:{port} <user>@<server>  then open {url}")
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
