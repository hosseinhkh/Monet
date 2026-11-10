"""Debug bundles: zip the logs of a training run / evaluation (optionally checkpoints and videos).

  python -m monet_action.bundle runs/d1_k10 -o d1_k10_debug.zip
  python -m monet_action.bundle runs/d1_k10 results/eval/d1_k10 -o d1_debug.zip --include-videos
  python -m monet_action.bundle runs/d2_lora -o d2_full.zip --include-checkpoints

What goes in (recursively, per source directory):
  * text/log files: .json .jsonl .md .txt .log .yaml .yml .csv .out .err, and the ``last`` pointer
  * small figures (.png/.jpg/.jpeg/.svg up to 10 MB)
  * ``ckpt_*/`` directories: only their text files (``meta.json``) unless ``include_checkpoints``
  * videos (.mp4/.webm/.gif/...) only with ``include_videos``
  * non-log files larger than ``max_file_mb`` (default 200 MB) are skipped unless
    ``include_checkpoints``; log-like files (.log/.txt/.out/.err/.jsonl) are never size-capped
  * hidden / temporary entries (``.tmp_ckpt_*``, ``.cache``) and symlinks that leave the source
    directory are never included
Every bundle carries ``bundle_manifest.json``: creation time, host, sources, included and
skipped files, git commit, python/package versions, a pip-freeze equivalent and
``nvidia-smi`` output when available. The same code backs the dashboard's download buttons.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import socket
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Sequence, Tuple, Union

MB = 1024 * 1024
TEXT_EXTS = {".json", ".jsonl", ".md", ".txt", ".log", ".yaml", ".yml", ".csv", ".out", ".err"}
LOG_EXTS = {".log", ".txt", ".out", ".err", ".jsonl"}  # never size-capped
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".svg"}
VIDEO_EXTS = {".mp4", ".webm", ".gif", ".avi", ".mov", ".mkv"}
WEIGHT_EXTS = {".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".npz", ".npy"}
STORED_EXTS = VIDEO_EXTS | WEIGHT_EXTS | {".png", ".jpg", ".jpeg", ".zip", ".gz"}  # already compressed / binary
POINTER_FILES = {"last"}
SKIP_DIRS = {"__pycache__", ".git"}
IMAGE_MAX_MB = 10
DEFAULT_MAX_FILE_MB = 200
MANIFEST_NAME = "bundle_manifest.json"
VERSION_PACKAGES = ("torch", "torchvision", "transformers", "tokenizers", "peft", "accelerate", "safetensors",
                    "numpy", "pyarrow", "huggingface-hub", "mujoco", "robosuite", "bddl", "gym", "monet_action")
# Only these environment variables are recorded (never dump the full environment: tokens/keys).
ENV_WHITELIST = ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "PYOPENGL_PLATFORM", "TOKENIZERS_PARALLELISM",
                 "TRANSFORMERS_VERBOSITY", "HF_HUB_OFFLINE", "SLURM_JOB_ID", "SLURM_JOB_NAME", "VENV")
MAX_SKIPPED_LISTED = 2000


@dataclass
class BundleEntry:
    path: Path  # absolute source path
    arcname: str  # path inside the zip
    size: int


@dataclass
class BundlePlan:
    entries: List[BundleEntry] = field(default_factory=list)
    skipped: List[Dict] = field(default_factory=list)
    n_skipped: int = 0

    def skip(self, arcname: str, size: int, reason: str) -> None:
        self.n_skipped += 1
        if len(self.skipped) < MAX_SKIPPED_LISTED:
            self.skipped.append({"path": arcname, "size": size, "reason": reason})


def classify(rel: Path, size: int, *, include_checkpoints: bool = False, include_videos: bool = False,
             max_file_mb: float = DEFAULT_MAX_FILE_MB) -> Tuple[bool, Optional[str]]:
    """Decide whether ``rel`` (path relative to a source dir) belongs in the bundle."""
    parts = rel.parts
    name, ext = rel.name, rel.suffix.lower()
    if any(p.startswith(".") for p in parts):
        return False, "hidden or temporary"
    in_ckpt = any(p.startswith("ckpt_") for p in parts[:-1])
    if ext in VIDEO_EXTS:
        return (True, None) if include_videos else (False, "video (enable include_videos)")
    if in_ckpt:
        if include_checkpoints or ext in TEXT_EXTS:
            return True, None
        return False, "checkpoint weights (enable include_checkpoints)"
    if name in POINTER_FILES:
        return True, None
    if ext in TEXT_EXTS:
        if ext not in LOG_EXTS and size > max_file_mb * MB and not include_checkpoints:
            return False, f"larger than {max_file_mb:g} MB"
        return True, None
    if ext in IMAGE_EXTS:
        return (True, None) if size <= IMAGE_MAX_MB * MB else (False, f"image larger than {IMAGE_MAX_MB} MB")
    if include_checkpoints and ext in WEIGHT_EXTS:
        return True, None
    return False, "not a log / text file"


def plan_bundle(sources: Sequence[Tuple[Union[str, Path], str]], *, include_checkpoints: bool = False,
                include_videos: bool = False, max_file_mb: float = DEFAULT_MAX_FILE_MB) -> BundlePlan:
    """``sources``: (directory, prefix inside the zip). Duplicate files (overlapping sources) are kept once."""
    plan = BundlePlan()
    seen = set()
    for src, prefix in sources:
        root = Path(src)
        if not root.is_dir():
            continue
        root_r = root.resolve()
        prefix = prefix.strip("/")
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                rel = p.relative_to(root)
                arc = f"{prefix}/{rel.as_posix()}" if prefix else rel.as_posix()
                try:
                    real = p.resolve()
                    if real != root_r and root_r not in real.parents:
                        plan.skip(arc, 0, "symlink outside the source directory")
                        continue
                    if not real.is_file():
                        continue
                    size = real.stat().st_size
                except OSError:
                    continue
                if real in seen:
                    continue
                ok, reason = classify(rel, size, include_checkpoints=include_checkpoints,
                                      include_videos=include_videos, max_file_mb=max_file_mb)
                if not ok:
                    plan.skip(arc, size, reason or "excluded")
                    continue
                seen.add(real)
                plan.entries.append(BundleEntry(real, arc, size))
    return plan


# ----------------------------------------------------------------------------- manifest
def _run(cmd: List[str], timeout: float = 10.0) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_info(paths: Sequence[Union[str, Path]]) -> Dict[str, Optional[object]]:
    candidates = [Path(p) for p in paths] + [Path(__file__).resolve().parent]
    for c in candidates:
        d = c if c.is_dir() else c.parent
        commit = _run(["git", "-C", str(d), "rev-parse", "HEAD"], 5)
        if commit:
            status = _run(["git", "-C", str(d), "status", "--porcelain"], 10)
            return {"commit": commit, "branch": _run(["git", "-C", str(d), "rev-parse", "--abbrev-ref", "HEAD"], 5),
                    "dirty": bool(status) if status is not None else None, "repo_dir": str(d)}
    return {"commit": None, "branch": None, "dirty": None, "repo_dir": None}


def package_versions() -> Dict[str, Optional[str]]:
    from importlib import metadata

    out = {}
    for name in VERSION_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def pip_freeze() -> List[str]:
    from importlib import metadata

    seen = {}
    for dist in metadata.distributions():
        try:
            name = dist.metadata["Name"]
        except Exception:  # pragma: no cover - broken metadata
            continue
        if name and name.lower() not in seen:
            seen[name.lower()] = f"{name}=={dist.version}"
    return sorted(seen.values(), key=str.lower)


def build_manifest(sources: Sequence[Tuple[Union[str, Path], str]], plan: BundlePlan, options: Dict,
                   extra: Optional[Dict] = None) -> Dict:
    now = time.time()
    manifest = {
        "created_at": _dt.datetime.fromtimestamp(now, _dt.timezone.utc).isoformat(),
        "created_at_unix": now,
        "host": socket.gethostname(),
        "source_dirs": [{"path": str(Path(s).resolve()), "prefix": p} for s, p in sources],
        "options": options,
        "n_files": len(plan.entries),
        "total_bytes": sum(e.size for e in plan.entries),
        "files": [{"path": e.arcname, "size": e.size} for e in plan.entries],
        "n_skipped": plan.n_skipped,
        "skipped": plan.skipped,
        "git": git_info([s for s, _ in sources]),
        "git_commit": None,
        "python": {"version": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "packages": package_versions(),
        "pip_freeze": pip_freeze(),
        "nvidia_smi": _run(["nvidia-smi"], 15),
        "env": {k: os.environ.get(k) for k in ENV_WHITELIST if os.environ.get(k) is not None},
    }
    manifest["git_commit"] = manifest["git"]["commit"]
    if extra:
        manifest["extra"] = extra
    return manifest


# ----------------------------------------------------------------------------- writing
def write_bundle(dest: Union[str, Path, BinaryIO], sources: Sequence[Tuple[Union[str, Path], str]], *,
                 include_checkpoints: bool = False, include_videos: bool = False,
                 max_file_mb: float = DEFAULT_MAX_FILE_MB, extra: Optional[Dict] = None) -> Dict:
    """Write a zip to ``dest`` (path or writable, possibly non-seekable, binary stream). Returns the manifest.

    Text files are snapshotted in memory before compression, so a log that is being appended to
    while the bundle is built is captured consistently up to that moment.
    """
    options = {"include_checkpoints": include_checkpoints, "include_videos": include_videos,
               "max_file_mb": max_file_mb}
    plan = plan_bundle(sources, **options)
    written: List[BundleEntry] = []
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
        for e in plan.entries:
            ext = e.path.suffix.lower()
            ctype = zipfile.ZIP_STORED if ext in STORED_EXTS else zipfile.ZIP_DEFLATED
            try:
                if ext in TEXT_EXTS or e.path.name in POINTER_FILES or e.size <= 4 * MB:
                    data = e.path.read_bytes()
                    e.size = len(data)
                    zi = zipfile.ZipInfo(e.arcname, date_time=time.localtime(e.path.stat().st_mtime)[:6])
                    zi.compress_type = ctype
                    zi.external_attr = 0o644 << 16
                    zf.writestr(zi, data)
                else:
                    zf.write(e.path, e.arcname, compress_type=ctype)
                written.append(e)
            except OSError as err:  # file vanished / unreadable while bundling
                plan.skip(e.arcname, e.size, f"unreadable: {err}")
        plan.entries = written
        manifest = build_manifest(sources, plan, options, extra)
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, default=str))
    return manifest


def _default_prefixes(dirs: Sequence[Path]) -> List[str]:
    names = [d.resolve().name or "root" for d in dirs]
    out = []
    for d, n in zip(dirs, names):
        out.append(f"{d.resolve().parent.name}/{n}" if names.count(n) > 1 else n)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Zip run/eval logs (optionally checkpoints/videos) for debugging.")
    ap.add_argument("dirs", nargs="+", help="run and/or evaluation directories")
    ap.add_argument("-o", "--output", default=None, help="output .zip (default: <name>_bundle_<time>.zip)")
    ap.add_argument("--include-checkpoints", action="store_true", help="also add ckpt_*/ weights")
    ap.add_argument("--include-videos", action="store_true", help="also add rollout videos")
    ap.add_argument("--max-file-mb", type=float, default=DEFAULT_MAX_FILE_MB,
                    help="skip single non-log files above this size (ignored with --include-checkpoints)")
    a = ap.parse_args(argv)
    dirs = [Path(d) for d in a.dirs]
    missing = [str(d) for d in dirs if not d.is_dir()]
    if missing:
        print(f"error: not a directory: {', '.join(missing)}", file=sys.stderr)
        return 2
    out = Path(a.output) if a.output else Path(
        f"{dirs[0].resolve().name}_bundle_{time.strftime('%Y%m%d-%H%M%S')}.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    sources = list(zip(dirs, _default_prefixes(dirs)))
    m = write_bundle(out, sources, include_checkpoints=a.include_checkpoints, include_videos=a.include_videos,
                     max_file_mb=a.max_file_mb)
    reasons: Dict[str, int] = {}
    for s in m["skipped"]:
        reasons[s["reason"]] = reasons.get(s["reason"], 0) + 1
    print(f"wrote {out} ({out.stat().st_size / MB:.2f} MB): {m['n_files']} files, "
          f"{m['total_bytes'] / MB:.2f} MB uncompressed; skipped {m['n_skipped']}")
    for r, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  skipped {n:5d}  {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
