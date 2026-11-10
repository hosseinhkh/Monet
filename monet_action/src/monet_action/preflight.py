"""Preflight checks run by scripts/preflight.sh before any long training run.

    python -m monet_action.preflight env [--allow-cpu]
    python -m monet_action.preflight data [--root data/libero_spatial_image]
    python -m monet_action.preflight weights
    python -m monet_action.preflight render
    python -m monet_action.preflight gpu-tests  --log results/preflight/06_gpu_tests.log
    python -m monet_action.preflight audit      --report results/audit/g0_report_monet7b.json
    python -m monet_action.preflight memory     --summary results/preflight/memory_profile.json
    python -m monet_action.preflight overfit    --run runs/smoke

Every check prints PASS / WARN / FAIL lines with the measured numbers and exits non-zero on FAIL.
"""
from __future__ import annotations

import argparse
import json
import os

from monet_action.runtime_env import apply_defaults, nvidia_smi_gpus

apply_defaults()  # before torch: GPU pinning (GPU_ID, default 0 / auto) and MUJOCO_GL, as scripts/common.sh
import shutil
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]
Result = Tuple[str, str, str]  # (status, name, detail)
MONET_REPO, MONET_REV = "NOVAglow646/Monet-7B", "39c1e320a3e15c357bff911fda263923edbdda37"
EXPECTED = {"transformers": "4.54.0", "mujoco": "3.3.2", "robosuite": "1.4.1", "peft": "0.21.2"}


def _ver(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "missing"


def _other_gpu_processes(gpu_index: str) -> str:
    """Compute processes on the physical GPU (nvidia-smi numbering) other than this one."""
    import subprocess

    try:
        uuid = subprocess.run(["nvidia-smi", "-i", gpu_index, "--query-gpu=uuid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""
    rows = [ln.split(", ") for ln in out.strip().splitlines() if ln.strip()]
    mine = str(os.getpid())
    return "; ".join(f"pid {p[1]} {p[2]} {p[3]}" for p in rows if len(p) >= 4 and p[0] == uuid and p[1] != mine)


def _report(results: List[Result]) -> int:
    for st, name, detail in results:
        print(f"  [{st:4s}] {name:34s} {detail}")
    fails = [r for r in results if r[0] == "FAIL"]
    print(f"  -> {'FAIL' if fails else 'PASS'} ({len(fails)} failed, {sum(r[0] == 'WARN' for r in results)} warnings)")
    return 1 if fails else 0


# ------------------------------------------------------------------------------------------ checks
def check_env(allow_cpu: bool = False) -> int:
    r: List[Result] = []
    py = sys.version_info
    r.append(("PASS" if (3, 10) <= py[:2] <= (3, 12) else "FAIL", "python", sys.version.split()[0]))
    for pkg, want in EXPECTED.items():
        got = _ver(pkg)
        r.append(("PASS" if got == want else ("FAIL" if got == "missing" else "WARN"), pkg, f"{got} (tested {want})"))
    import torch

    r.append(("PASS" if torch.__version__.startswith("2.7") else "WARN", "torch", f"{torch.__version__} cuda={torch.version.cuda}"))
    try:
        from libero.libero import benchmark  # noqa: F401

        cfg = Path.home() / ".libero" / "config.yaml"
        r.append(("PASS" if cfg.exists() else "FAIL", "LIBERO import + ~/.libero/config.yaml", str(cfg)))
    except Exception as e:  # pragma: no cover
        r.append(("FAIL", "LIBERO import", repr(e)))
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        mem = p.total_memory / 1024 ** 3
        r.append(("PASS", "GPU", f"{p.name} sm_{p.major}{p.minor} {mem:.0f} GB, {torch.cuda.device_count()} device(s)"))
        r.append(("PASS" if mem >= 80 else "WARN", "GPU memory >= 80 GB", f"{mem:.0f} GB (D2 batch size must then be profiled)"))
        try:
            a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
            for _ in range(3):  # warm-up: cuBLAS handle + kernel selection are not throughput
                b = a @ a
            torch.cuda.synchronize()
            n, t0 = 30, time.time()
            for _ in range(n):
                b = a @ a
            torch.cuda.synchronize()
            tflops = n * 2 * 8192 ** 3 / (time.time() - t0) / 1e12
            ok = bool(torch.isfinite(b).all())
            del a, b
            torch.cuda.empty_cache()
            r.append(("FAIL" if not ok else ("PASS" if tflops >= 50 else "WARN"), "bf16 matmul",
                      f"finite={ok}, {tflops:.0f} TFLOP/s" + ("" if tflops >= 50 else " (low: GPU shared with other jobs or throttled?)")))
        except Exception as e:
            r.append(("FAIL", "bf16 matmul", repr(e)))
        free, total = torch.cuda.mem_get_info()
        free_gb = free / 1024 ** 3
        # Monet-7B bf16 weights 16.6 GB + activations: D0/D1 need ~30 GB, D2 more (see the memory step)
        need = float(os.environ.get("MONET_MIN_FREE_GB", "40"))
        cards = nvidia_smi_gpus()
        sel = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        if cards:
            table = "; ".join(f"{'*' if str(i) == sel else ' '}GPU {i}: {fr:.0f}/{tot:.0f} GB free" for i, _, fr, tot in cards)
            r.append(("PASS", "GPUs on this machine (* = selected)", table))
        best = max(cards, key=lambda c: c[2]) if cards else None
        hint = ""
        if best is not None and str(best[0]) != sel and best[2] > free_gb + 5:
            hint = f" -> GPU {best[0]} has {best[2]:.0f} GB free: run with GPU_ID={best[0]} (or GPU_ID=auto)"
        status = "PASS" if free / total > 0.9 else ("WARN" if free_gb >= need else "FAIL")
        r.append((status, f"selected GPU free memory (need >= {need:.0f} GB)", f"{free_gb:.0f}/{total / 1024 ** 3:.0f} GB free{hint}"))
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        r.append(("PASS" if torch.cuda.device_count() == 1 else "WARN", "pinned to one GPU",
                  f"CUDA_VISIBLE_DEVICES={vis} MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID')} "
                  f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')} -> {torch.cuda.device_count()} visible"))
        others = _other_gpu_processes((vis or "0").split(",")[0])
        r.append(("PASS" if not others else "WARN", "other processes on this GPU", others or "none"))
        frac = float(os.environ.get("MONET_GPU_RESERVE", "0") or 0)
        if frac > 0:
            from .gpu import reserve_gpu_memory

            got = reserve_gpu_memory(frac)
            held = torch.cuda.memory_reserved(0) / 1024 ** 3
            r.append(("PASS" if held >= 0.8 * frac * total / 1024 ** 3 else "WARN", "MONET_GPU_RESERVE works",
                      f"fraction {frac}: this process could hold {held:.0f} GB (+{got:.0f} GB)"))
            torch.cuda.empty_cache()
    else:
        r.append(("WARN" if allow_cpu else "FAIL", "CUDA available", "no CUDA device"))
    disk = shutil.disk_usage(ROOT).free / 1024 ** 3
    low = "WARN" if allow_cpu else "FAIL"  # CPU self-test machines do not hold the full data + weights
    r.append(("PASS" if disk >= 60 else ("WARN" if disk >= 30 else low), "disk free (repo)", f"{disk:.0f} GB (need ~25 GB data + checkpoints)"))
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hf.mkdir(parents=True, exist_ok=True)
    hfree = shutil.disk_usage(hf).free / 1024 ** 3
    r.append(("PASS" if hfree >= 20 else low, "disk free (HF cache)", f"{hfree:.0f} GB at {hf} (Monet-7B = 16.6 GB)"))
    try:
        with open("/proc/meminfo") as f:
            ram = float(f.readline().split()[1]) / 1024 ** 2
        r.append(("PASS" if ram >= 48 else "WARN", "RAM", f"{ram:.0f} GB (dataset is held in RAM: ~13 GB peak while loading)"))
    except Exception:
        pass
    r.append(("PASS" if os.environ.get("MUJOCO_GL") in ("egl", "osmesa") else "WARN", "MUJOCO_GL", str(os.environ.get("MUJOCO_GL"))))
    return _report(r)


def check_data(root: str, partial: bool = False) -> int:
    """``partial``: count mismatches are warnings (machines holding only a few shards)."""
    r: List[Result] = []
    bad = "WARN" if partial else "FAIL"
    p = Path(root)
    info_p = p / "meta" / "info.json"
    if not info_p.exists():
        return _report([("FAIL", "dataset", f"{info_p} missing - run scripts/download_data.sh")])
    info = json.loads(info_p.read_text())
    r.append(("PASS" if info.get("total_episodes") == 432 else "FAIL", "episodes (meta)", str(info.get("total_episodes"))))
    r.append(("PASS" if info.get("total_frames") == 52970 else "FAIL", "frames (meta)", str(info.get("total_frames"))))
    files = sorted((p / "data").glob("chunk-*/file-*.parquet"))
    size = sum(f.stat().st_size for f in files) / 1024 ** 3
    r.append(("PASS" if len(files) == 69 else bad, "parquet files", f"{len(files)} files, {size:.1f} GB (expected 69, ~6.5 GB)"))
    try:
        import pyarrow.parquet as pq

        n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        r.append(("PASS" if n == 52970 else bad, "frames (parquet rows)", str(n)))
    except Exception as e:
        r.append(("FAIL", "parquet readable", repr(e)))
    from .libero_dataset import LeRobotLiberoFrames

    fr = LeRobotLiberoFrames(root, file_limit=1)
    e = fr.episodes[0]
    img = fr.image("image", e.start)
    r.append(("PASS" if img.shape == (256, 256, 3) else "FAIL", "decode one frame", f"{img.shape}, instruction: {e.instruction[:50]}..."))
    return _report(r)


def check_weights() -> int:
    r: List[Result] = []
    path = os.environ.get("MONET7B_PATH")
    try:
        if not path:
            from huggingface_hub import snapshot_download

            path = snapshot_download(MONET_REPO, revision=MONET_REV, local_files_only=True)
        st = sorted(Path(path).glob("model-*.safetensors"))
        gb = sum(f.stat().st_size for f in st) / 1024 ** 3
        r.append(("PASS" if len(st) == 4 and gb > 15 else "FAIL", "Monet-7B safetensors", f"{len(st)} shards, {gb:.1f} GB at {path}"))
        cfg = json.loads((Path(path) / "config.json").read_text())
        ids = (cfg.get("latent_token_id"), cfg.get("latent_start_id"), cfg.get("latent_end_id"))
        r.append(("PASS" if ids == (151665, 151666, 151667) else "FAIL", "latent token ids", str(ids)))
        from .config import MonetConfig
        from .monet_loader import load_processor

        proc = load_processor(MonetConfig(), path)
        r.append(("PASS", "tokenizer/processor", f"{type(proc).__name__}, vocab {len(proc.tokenizer)}"))
    except Exception as e:
        r.append(("FAIL", "Monet-7B weights", f"{e!r} - run scripts/download_data.sh"))
    return _report(r)


def check_render() -> int:
    from .libero_env import LiberoEnv

    r: List[Result] = []
    t0 = time.time()
    env = LiberoEnv("libero_spatial", 0, camera_size=256, seed=0, jpeg_quality=90)
    try:
        o = env.reset(0)
        t1 = time.time()
        for _ in range(10):
            st = env.step([0.2, 0, 0, 0, 0, 0, -1])
        step_ms = (time.time() - t1) / 10 * 1000
    finally:
        env.close()
    ok = o.images["image"].std() > 5 and o.images["wrist_image"].std() > 5
    r.append(("PASS" if ok else "FAIL", f"render ({os.environ.get('MUJOCO_GL')})", f"images non-blank={ok}, env build+reset {t1 - t0:.1f}s"))
    r.append(("PASS" if step_ms < 200 else "WARN", "env.step incl. rendering", f"{step_ms:.0f} ms/step"))
    r.append(("PASS" if not (st.observation.state == o.state).all() else "FAIL", "observation refresh", "state changes after env.step"))
    return _report(r)


def check_gpu_tests(log: str) -> int:
    t = Path(log).read_text() if Path(log).exists() else ""
    last = [ln for ln in t.splitlines() if "passed" in ln or "failed" in ln or "skipped" in ln or "error" in ln]
    summary = last[-1] if last else "no pytest summary"
    import re

    counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|skipped|errors?|xfailed|xpassed)", summary)}
    bad = counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0) + counts.get("skipped", 0)
    ok = counts.get("passed", 0) > 0 and bad == 0  # xfailed (soft sanity checks) is allowed
    rep = ROOT / "results" / "audit" / "monet7b_gpu_tests.json"
    detail = summary.strip()
    if rep.exists():
        d = json.loads(rep.read_text())
        detail += f" | parity {d.get('parity', {})!s:.120} | d2 batch-1 peak {d.get('d2_step_peak_mem_gb_batch1', 'n/a')}"
    return _report([("PASS" if ok else "FAIL", "real Monet-7B GPU tests (must not skip)", detail)])


def check_audit(report: str) -> int:
    p = Path(report)
    if not p.exists():
        return _report([("FAIL", "audit report", f"{p} missing")])
    d = json.loads(p.read_text())
    r: List[Result] = []
    m = d.get("monet", {})
    r.append(("PASS" if not m.get("load_info", {}).get("missing_keys") else "FAIL", "no missing weights", str(m.get("load_info", {}).get("missing_keys", [])[:3])))
    for conv in ("vllm", "sft"):
        c = m.get("conventions", {}).get(conv, {})
        errs = [x.get("rel_err", 1.0) for x in c.get("parity", [])]
        r.append(("PASS" if c.get("latents_finite") else "FAIL", f"{conv}: latents finite", str(c.get("latents_shape"))))
        r.append(("PASS" if errs and max(errs) < 5e-2 else "FAIL", f"{conv}: parity vs authors' code", f"max rel err {max(errs) if errs else 'n/a'}"))
        r.append(("PASS" if c.get("state_perturbation_latent_rel_change", 0) > 1e-3 else "FAIL", f"{conv}: state dependence",
                  f"rel change {c.get('state_perturbation_latent_rel_change')}"))
    s = d.get("sim", {})
    corr = s.get("image_corr_with_dataset", {}).get("policy_view_rot180", 0)
    r.append(("PASS" if corr > 0.8 else "FAIL", "sim image orientation vs dataset", f"corr {corr:.3f}"))
    r.append(("PASS" if s.get("observation_refreshed") else "FAIL", "sim observation refresh", str(s.get("observation_refreshed"))))
    dd = d.get("data", {})
    r.append(("PASS" if dd.get("train_val_disjoint") else "FAIL", "train/val episode-disjoint", str(dd.get("train_val_disjoint"))))
    return _report(r)


def check_memory(summary: str, budget_gb: float = 85.0) -> int:
    p = Path(summary)
    if not p.exists():
        return _report([("FAIL", "memory profile", f"{p} missing")])
    rows = json.loads(p.read_text())
    r: List[Result] = []
    fit = [x for x in rows if x.get("ok") and (x.get("peak_gb") or 1e9) <= budget_gb]
    for x in rows:
        r.append(("PASS" if x in fit else "WARN", f"micro-batch {x['batch_size']}",
                  f"peak {x.get('peak_gb')} GB, {x.get('step_time_s')} s/step" + ("" if x.get("ok") else " (failed / OOM)")))
    if not fit:
        r.append(("FAIL", "recommended micro-batch", f"nothing fits in {budget_gb} GB - use monet.recurrence_impl=recompute monet.gradient_checkpointing=true"))
    else:
        b = max(x["batch_size"] for x in fit)
        accum = max(1, 16 // b)
        r.append(("PASS", "recommended micro-batch", f"train.batch_size={b} train.grad_accum={accum} (effective {b * accum})"))
        (p.parent / "recommended_batch.txt").write_text(f"train.batch_size={b} train.grad_accum={accum}\n")
    return _report(r)


def check_overfit(run: str) -> int:
    d = Path(run)
    r: List[Result] = []
    st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
    r.append(("PASS" if st.get("state") == "finished" else "FAIL", "smoke run finished", str(st.get("state"))))
    ga = json.loads((d / "gradient_audit.json").read_text()) if (d / "gradient_audit.json").exists() else {}
    r.append(("PASS" if ga.get("passed") else "FAIL", "gradient audit", ", ".join(f"{k}={v['grad_norm']:.2e}" for k, v in ga.get("report", {}).items())))
    ov = [json.loads(x) for x in (d / "overfit_log.jsonl").read_text().splitlines()] if (d / "overfit_log.jsonl").exists() else []
    fm = [x["fm_loss"] for x in ov]
    mono = len(fm) >= 3 and sum(b < a for a, b in zip(fm, fm[1:])) >= len(fm) - 2
    r.append(("PASS" if mono else "FAIL", "fixed-noise FM loss decreases", " -> ".join(f"{v:.3f}" for v in fm)))
    r.append(("PASS" if fm and fm[-1] < 0.5 * fm[0] else "FAIL", "final < 0.5 x initial", f"{fm[-1]:.3f} vs {fm[0]:.3f}" if fm else "n/a"))
    if ov:
        r.append(("PASS" if ov[-1]["gripper_acc"] > 0.9 else "WARN", "gripper sign accuracy", f"{ov[-1]['gripper_acc']:.3f}"))
    return _report(r)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["env", "data", "weights", "render", "gpu-tests", "audit", "memory", "overfit"])
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--partial", action="store_true", help="data: only some shards downloaded (CPU self-test)")
    ap.add_argument("--root", default=str(ROOT / "data" / "libero_spatial_image"))
    ap.add_argument("--log", default=str(ROOT / "results" / "preflight" / "06_gpu_tests.log"))
    ap.add_argument("--report", default=str(ROOT / "results" / "audit" / "g0_report_monet7b.json"))
    ap.add_argument("--summary", default=str(ROOT / "results" / "preflight" / "memory_profile.json"))
    ap.add_argument("--budget-gb", type=float, default=85.0)
    ap.add_argument("--run", default=str(ROOT / "runs" / "smoke"))
    a = ap.parse_args(argv)
    print(f"== preflight: {a.check}")
    fn = {"env": lambda: check_env(a.allow_cpu), "data": lambda: check_data(a.root, a.partial), "weights": check_weights,
          "render": check_render, "gpu-tests": lambda: check_gpu_tests(a.log), "audit": lambda: check_audit(a.report),
          "memory": lambda: check_memory(a.summary, a.budget_gb), "overfit": lambda: check_overfit(a.run)}[a.check]
    sys.exit(fn())


if __name__ == "__main__":
    main()
