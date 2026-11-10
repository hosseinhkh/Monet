"""Monitoring dashboard + debug bundles: API shapes, incremental series, robustness, security, zips.

Everything runs against a synthetic run / eval directory that follows the documented file contract
(docs/DASHBOARD.md) and an in-process server on a free port. No network access is needed.
"""
import http.client
import io
import json
import os
import threading
import time
import zipfile
from pathlib import Path

import pytest

from monet_action.bundle import MANIFEST_NAME, write_bundle
from monet_action.bundle import main as bundle_main
from monet_action.dashboard.server import JsonlIndex, downsample, make_server, safe_join


# ----------------------------------------------------------------------------- fixtures
def _jsonl(path: Path, rows, partial_tail: str = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
        if partial_tail is not None:
            f.write(partial_tail)  # a line still being written (no trailing newline)


def make_fake_tree(tmp: Path):
    now = time.time()
    runs, results = tmp / "runs", tmp / "results"
    run = runs / "d1_test"
    cfg = {"name": "d1_test", "variant": "D1", "description": "frozen Monet, K=10",
           "monet": {"model_path": "NOVAglow646/Monet-7B", "num_latents": 10, "latent_convention": "sft",
                     "lora_enabled": False, "reasoning_mode": "self_fed", "recurrence_impl": "cached"},
           "conditioner": {"feature_source": "vl_plus_latents"}, "action_head": {"type": "dit", "action_horizon": 8},
           "train": {"max_steps": 100, "batch_size": 8, "grad_accum": 2, "lr_head": 1e-4, "output_dir": "runs/d1_test"}}
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(cfg))
    (run / "status.json").write_text(json.dumps({
        "kind": "train", "name": "d1_test", "variant": "D1", "state": "running", "pid": 123, "host": "gpu-box",
        "started_at": now - 600, "updated_at": now, "step": 40, "max_steps": 100, "samples": 640, "eta_s": 900.0,
        "last_loss": 0.5, "error": None, "output_dir": str(run)}))
    train_rows = []
    for i in range(1, 31):
        train_rows.append({"step": i, "loss": 1.0 / i, "fm_loss": 1.0 / i, "grad_norm": 1.0, "lr_scale": 0.5,
                           "lrs": {"head": 1e-4, "state": 1e-4}, "step_time_s": 0.2, "data_time_s": 0.01,
                           "peak_mem_gb": float("nan"), "samples": 8 * i, "time": now - 30 + i,
                           "grad_groups": {"action_head": {"grad_norm": 2.0, "finite": True, "n_tensors": 3,
                                                           "n_tensors_nonzero_grad": 3}} if i == 1 else None})
        if i == 10:
            train_rows.append({"event": "resume", "step": 10, "time": now})
    train_rows.insert(15, "{this line is malformed json")
    _jsonl(run / "train_log.jsonl", train_rows, partial_tail='{"step": 31, "loss": 0.03')
    _jsonl(run / "val_log.jsonl", [{"step": s, "fm_loss": 0.5 / s, "action_mse": 0.4 / s, "action_mse_first": 0.3 / s,
                                    "gripper_acc": 0.9, "val_batches": 4, "time": now} for s in (10, 20, 30)])
    _jsonl(run / "overfit_log.jsonl", [{"step": 1, "fm_loss": 1.5, "action_mse": 1.6, "gripper_acc": 0.5}])
    _jsonl(run / "system.jsonl", [{"time": now - 60 + 10 * i, "gpus": [{"index": 0, "name": "NVIDIA RTX PRO 6000",
                                                                       "util_percent": 90.0, "mem_used_gb": 60.0,
                                                                       "mem_total_gb": 96.0, "temp_c": 70, "power_w": 400}],
                                   "cpu_percent": 30.0, "ram_used_gb": 40.0, "ram_total_gb": 256.0,
                                   "disk_free_gb": 500.0, "torch_mem_allocated_gb": None,
                                   "torch_max_mem_allocated_gb": None} for i in range(5)])
    _jsonl(run / "events.jsonl", [{"time": now - 600, "event": "start"}, {"time": now - 300, "event": "checkpoint", "step": 20}])
    for name, obj in [("param_report.json", {"monet_frozen": 8_000_000_000, "trainable_total": 25_000_000}),
                      ("gradient_audit.json", {"step": 0, "passed": True, "report": {"action_head": {"grad_norm": 2.0, "finite": True, "n_tensors": 3, "n_tensors_nonzero_grad": 3}}}),
                      ("memory_profile.json", {"peak_gb_after_3_steps": float("nan")}),
                      ("representation.json", {"K": 10, "latent_layer": -1, "pooling": "none", "example_prompt": "<|im_start|>..."}),
                      ("environment.json", {"torch": "2.7.1", "python": "3.11"}),
                      ("split.json", {"train_episodes": list(range(10)), "val_episodes": [10, 11]})]:
        (run / name).write_text(json.dumps(obj))
    (run / "logs").mkdir()
    (run / "logs" / "train.log").write_text("".join(f"line {i} ok\n" for i in range(100)) + "Traceback: boom error\n")
    ck = run / "ckpt_0000010"
    ck.mkdir()
    (ck / "trainable.pt").write_bytes(os.urandom(1024))
    (ck / "optimizer.pt").write_bytes(os.urandom(1024))
    (ck / "meta.json").write_text(json.dumps({"step": 10, "consumed_samples": 160}))
    (run / "last").write_text("ckpt_0000010")
    (run / ".tmp_ckpt_0000020").mkdir()
    (run / ".tmp_ckpt_0000020" / "trainable.pt").write_bytes(b"x" * 10)

    stale = runs / "stale_run"
    stale.mkdir()
    (stale / "config.json").write_text(json.dumps({"name": "stale_run", "train": {"max_steps": 50}}))
    (stale / "status.json").write_text(json.dumps({"kind": "train", "state": "running", "updated_at": now - 3600, "step": 5, "max_steps": 50}))
    legacy = runs / "legacy_run"  # older run: config + log only, like runs/overfit
    legacy.mkdir()
    (legacy / "config.json").write_text(json.dumps({"name": "legacy", "train": {"max_steps": 2}, "monet": {"num_latents": 0}}))
    _jsonl(legacy / "train_log.jsonl", [{"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.5}])
    (runs / "not_a_run").mkdir()
    (runs / "not_a_run" / "notes.txt").write_text("nothing")

    ev = results / "eval" / "d1_test"
    ev.mkdir(parents=True)
    (ev / "eval_config.json").write_text(json.dumps({"eval": {"seeds": [0], "episodes_per_task": 2, "task_ids": [0]},
                                                     "model_config": cfg}))
    eps = [{"task_id": 0, "init_state_idx": i, "seed": 0, "success": i == 0, "env_steps": 50, "action_calls": 25,
            "sim_time_ms": 1000.0, "call_latency_ms": [{"monet": 20.0, "action_head": 5.0}] * 3, "batch_sizes": [2, 2, 1],
            "wall_s": 3.0, "failure_tags": [] if i == 0 else ["gripper_timing"]} for i in range(2)]
    _jsonl(ev / "episodes.jsonl", eps)
    (ev / "summary.json").write_text(json.dumps({
        "n_episodes": 2, "success_rate": 0.5, "success_ci95": [0.09, 0.9], "gpu_seconds_per_success": float("inf"),
        "gpu_seconds_per_episode": 0.1, "per_task": {"0": {"successes": 1, "trials": 2, "rate": 0.5, "ci95": [0.09, 0.9]}},
        "latency_ms_per_call": {"total": {"p50": 25.0, "p95": 30.0, "mean": 25.0, "n": 6}}, "task_names": {"0": "pick"},
        "peak": float("nan")}))
    (ev / "traces").mkdir()
    _jsonl(ev / "traces" / "seed0_task0_init0.jsonl", [{"episode": {}, "trace": []}])
    (ev / "videos").mkdir()
    (ev / "videos" / "seed0_task0_init0_success.mp4").write_bytes(bytes(range(256)) * 4)
    (ev / "logs").mkdir()
    (ev / "logs" / "eval.log").write_text("eval started\n")
    (results / "eval" / "pending_eval").mkdir()  # an eval that has not written anything yet
    (results / "eval" / "pending_eval" / "status.json").write_text(json.dumps({"kind": "eval", "state": "running", "updated_at": now,
                                                                              "episodes_done": 0, "episodes_total": 300}))
    (results / "audit").mkdir()
    (results / "audit" / "g0_report_tiny.md").write_text("# G0\nok\n")
    (results / "audit" / "g0_report_tiny.json").write_text('{"a": NaN}')
    (results / "phase1_report.md").write_text("# report\n")
    (tmp / "secret.txt").write_text("TOP SECRET")
    return runs, results


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return make_fake_tree(tmp_path_factory.mktemp("dash"))


def _start(runs, results, token=None):
    srv = make_server(str(runs), str(results), host="127.0.0.1", port=0, token=token, quiet=True)
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    th.start()
    return srv


@pytest.fixture(scope="module")
def server(tree):
    srv = _start(*tree)
    yield srv
    srv.shutdown()
    srv.server_close()


def request(srv, path, headers=None, method="GET"):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    conn.request(method, path, headers=headers or {})
    r = conn.getresponse()
    body = r.read()
    conn.close()
    return r.status, dict(r.getheaders()), body


def get_json(srv, path, **kw):
    status, headers, body = request(srv, path, **kw)
    assert headers.get("Content-Type", "").startswith("application/json"), (path, status, body[:200])
    # strict JSON: NaN / Infinity must have been sanitised by the server
    return status, json.loads(body, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c} in {path}"))


# ----------------------------------------------------------------------------- unit helpers
def test_safe_join(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f.txt").write_text("x")
    assert safe_join(tmp_path, "a/f.txt") == (tmp_path / "a" / "f.txt").resolve()
    for bad in ("../x", "a/../../x", "/etc/passwd", "a/\x00", "..\\..\\x"):
        assert safe_join(tmp_path / "a", bad) is None or safe_join(tmp_path / "a", bad) == (tmp_path / "a").resolve() / "etc" / "passwd"
    os.symlink("/etc", tmp_path / "a" / "link")
    assert safe_join(tmp_path / "a", "link/passwd") is None  # symlink escaping the root


def test_downsample_keeps_bounds():
    rows = list(range(5000))
    out, ds = downsample(rows, 2000)
    assert ds and len(out) <= 2000 and out[0] == 0 and out[-1] == 4999
    assert downsample(rows[:10], 2000) == (rows[:10], False)


def test_jsonl_index_incremental_and_partial_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    _jsonl(p, [{"i": 0}, {"i": 1}, "garbage{"], partial_tail='{"i": 3')
    idx = JsonlIndex()
    r = idx.read(p, 0)
    assert [x["i"] for x in r["rows"]] == [0, 1] and r["next"] == 3  # malformed counted, partial not
    with open(p, "a") as f:
        f.write('}\n{"i": 4}\n')
    r2 = idx.read(p, r["next"])
    assert [x["i"] for x in r2["rows"]] == [3, 4] and r2["next"] == 5 and not r2["reset"]
    p.write_text('{"i": 9}\n')  # rewritten shorter -> cursor beyond end -> reset
    r3 = idx.read(p, r2["next"])
    assert r3["reset"] and [x["i"] for x in r3["rows"]] == [9]


# ----------------------------------------------------------------------------- API
def test_static_and_config(server):
    st, hd, body = request(server, "/")
    assert st == 200 and b"static/app.js" in body and "Content-Security-Policy" in hd
    st, hd, body = request(server, "/static/app.js")
    assert st == 200 and hd["Content-Type"].startswith("text/javascript")
    assert request(server, "/static/../server.py")[0] in (403, 404)
    st, cfg = get_json(server, "/api/config")
    assert st == 200 and cfg["refresh"] == 10 and cfg["token_required"] is False


def test_runs_listing(server):
    st, d = get_json(server, "/api/runs")
    runs = {r["name"]: r for r in d["runs"]}
    assert set(runs) == {"d1_test", "stale_run", "legacy_run"}  # not_a_run has no config/status
    r = runs["d1_test"]
    assert r["state"] == "running" and not r["stale"] and r["progress"] == pytest.approx(0.4)
    assert r["K"] == 10 and r["lora"] is False and r["variant"] == "D1" and r["eta_s"] == 900.0
    assert runs["stale_run"]["display_state"] == "stale" and runs["stale_run"]["stale"]
    leg = runs["legacy_run"]
    assert leg["state_inferred"] and leg["state"] == "finished" and leg["last_loss"] == 0.5


def test_run_detail(server):
    st, d = get_json(server, "/api/run/d1_test")
    assert st == 200
    for k in ("summary", "status", "config", "key_config", "param_report", "gradient_audit", "memory_profile",
              "representation", "environment", "split", "checkpoints", "logs", "series", "files"):
        assert k in d, k
    assert d["memory_profile"]["peak_gb_after_3_steps"] is None  # NaN -> null
    assert d["checkpoints"][0]["name"] == "ckpt_0000010" and d["checkpoints"][0]["is_last"]
    assert d["split"] == {"train_episodes": 10, "val_episodes": 2}
    assert [lg["path"] for lg in d["logs"]] == ["logs/train.log"]
    assert d["series"]["train_log"]["exists"] and not d["series"]["sim_eval_log"]["exists"]
    assert all(not f["path"].startswith(".tmp") for f in d["files"])
    assert get_json(server, "/api/run/nope")[0] == 404


def test_series_incremental(server, tree):
    st, d = get_json(server, "/api/run/d1_test/series?file=train_log&since=0")
    rows = d["rows"]
    steps = [r["step"] for r in rows if "loss" in r]
    assert steps == list(range(1, 31))  # malformed line skipped, partial last line held back
    assert any(r.get("event") == "resume" for r in rows)
    assert d["next"] == 32 and rows[0]["peak_mem_gb"] is None
    st, d2 = get_json(server, f"/api/run/d1_test/series?file=train_log&since={d['next']}")
    assert d2["rows"] == [] and d2["next"] == 32
    with open(tree[0] / "d1_test" / "train_log.jsonl", "a") as f:  # finish the partial line, add one more
        f.write('}\n{"step": 32, "loss": 0.02}\n')
    st, d3 = get_json(server, f"/api/run/d1_test/series?file=train_log&since={d['next']}")
    assert [r["step"] for r in d3["rows"]] == [31, 32] and d3["next"] == 34 and d3["file_id"] == d["file_id"]
    for f in ("val_log", "overfit_log", "system", "events"):
        st, s = get_json(server, f"/api/run/d1_test/series?file={f}")
        assert st == 200 and s["exists"] and s["rows"], f
    st, s = get_json(server, "/api/run/d1_test/series?file=sim_eval_log")
    assert st == 200 and not s["exists"] and s["rows"] == []
    assert get_json(server, "/api/run/d1_test/series?file=../../secret")[0] == 400


def test_series_downsampling(server, tree):
    big = tree[0] / "big_run"
    big.mkdir()
    (big / "config.json").write_text("{}")
    _jsonl(big / "train_log.jsonl", [{"step": i, "loss": 1.0} for i in range(5000)])
    time.sleep(2.1)  # discovery cache TTL
    st, d = get_json(server, "/api/run/big_run/series?file=train_log")
    assert d["downsampled"] and len(d["rows"]) <= 2000 and d["rows"][-1]["step"] == 4999 and d["next"] == 5000


def test_log_tail_and_search(server):
    st, d = get_json(server, "/api/run/d1_test/log?file=logs/train.log&tail=10")
    assert st == 200 and len(d["lines"]) == 10 and d["lines"][-1].startswith("Traceback")
    st, d = get_json(server, "/api/run/d1_test/log?file=train.log&tail=5&q=LINE%209")
    assert all("line 9" in ln for ln in d["lines"]) and d["n_matched"] >= 1
    assert get_json(server, "/api/run/d1_test/log?file=../../../secret.txt")[0] == 403
    assert get_json(server, "/api/run/d1_test/log?file=config.json")[0] == 404  # not a log file


def test_evals(server):
    st, d = get_json(server, "/api/evals")
    evs = {e["name"]: e for e in d["evals"]}
    assert "eval/d1_test" in evs and "eval/pending_eval" in evs
    e = evs["eval/d1_test"]
    assert e["success_rate"] == 0.5 and e["gpu_seconds_per_success"] == "Infinity" and e["run_name"] == "d1_test"
    assert e["n_success"] == 1 and e["latency"]["total"]["p50"] == 25.0
    assert evs["eval/pending_eval"]["state"] == "running" and evs["eval/pending_eval"]["episodes_total"] == 300
    st, det = get_json(server, "/api/eval/eval%2Fd1_test")
    assert st == 200 and len(det["episodes"]) == 2
    ep0, ep1 = det["episodes"]
    assert ep0["video"] == "videos/seed0_task0_init0_success.mp4" and ep0["trace"] == "traces/seed0_task0_init0.jsonl"
    assert ep1["failure_tags"] == ["gripper_timing"] and ep0["call_ms_mean"] == pytest.approx(25.0)
    assert det["summary"]["peak"] is None and det["videos"] == ["videos/seed0_task0_init0_success.mp4"]
    st, lg = get_json(server, "/api/eval/eval%2Fd1_test/log")
    assert lg["lines"] == ["eval started"]


def test_compare_and_audit(server):
    st, d = get_json(server, "/api/compare?runs=d1_test,legacy_run,missing_run")
    assert [r["name"] for r in d["runs"]] == ["d1_test", "legacy_run"] and d["missing"] == ["missing_run"]
    tl = d["runs"][0]["series"]["train_log"]
    assert tl and all("event" not in r for r in tl) and d["runs"][0]["series"]["val_log"]
    assert [e["name"] for e in d["evals"]] == ["eval/d1_test"]
    st, a = get_json(server, "/api/audit")
    assert {f["path"] for f in a["audit"]} == {"audit/g0_report_tiny.md", "audit/g0_report_tiny.json"}
    assert [f["path"] for f in a["reports"]] == ["phase1_report.md"]


def test_files_types_ranges_and_download(server):
    st, hd, body = request(server, "/files/runs/d1_test/logs/train.log")
    assert st == 200 and hd["Content-Type"].startswith("text/plain") and body.startswith(b"line 0")
    assert "sandbox" in hd["Content-Security-Policy"] and hd["X-Content-Type-Options"] == "nosniff"
    st, hd, body = request(server, "/files/results/eval/d1_test/videos/seed0_task0_init0_success.mp4")
    assert st == 200 and hd["Content-Type"] == "video/mp4" and len(body) == 1024
    st, hd, body = request(server, "/files/results/eval/d1_test/videos/seed0_task0_init0_success.mp4", {"Range": "bytes=10-19"})
    assert st == 206 and body == bytes(range(10, 20)) and hd["Content-Range"] == "bytes 10-19/1024"
    st, hd, body = request(server, "/files/results/eval/d1_test/videos/seed0_task0_init0_success.mp4", {"Range": "bytes=-4"})
    assert st == 206 and body == bytes(range(252, 256))
    st, hd, _ = request(server, "/files/results/eval/d1_test/videos/seed0_task0_init0_success.mp4", {"Range": "bytes=5000-"})
    assert st == 416
    st, hd, _ = request(server, "/files/runs/d1_test/config.json?download=1")
    assert st == 200 and "attachment" in hd["Content-Disposition"]
    st, hd, _ = request(server, "/files/runs/d1_test/logs/train.log", method="HEAD")
    assert st == 200 and int(hd["Content-Length"]) > 0


@pytest.mark.parametrize("path", [
    "/files/runs/../secret.txt", "/files/runs/../../secret.txt", "/files/runs/%2e%2e/secret.txt",
    "/files/runs/%2e%2e%2fsecret.txt", "/files/runs/d1_test/..%2f..%2fsecret.txt", "/files/runs/%252e%252e/secret.txt",
    "/files/runs//etc/passwd", "/files/runs/%2Fetc%2Fpasswd", "/files/runs/..%5c..%5csecret.txt",
    "/files/other/secret.txt", "/files/runs/d1_test/%00.log", "/files/runs/../../../../../../etc/passwd",
])
def test_path_traversal_rejected(server, path):
    st, hd, body = request(server, path)
    assert st in (403, 404), (path, st)
    assert b"TOP SECRET" not in body and b"root:" not in body


def test_token_enforcement(tree):
    srv = _start(*tree, token="s3cr3t")
    try:
        assert request(srv, "/api/runs")[0] == 401
        assert request(srv, "/")[0] == 401
        assert request(srv, "/api/runs?token=wrong")[0] == 401
        st, hd, _ = request(srv, "/api/runs?token=s3cr3t")
        assert st == 200 and "monet_dash_session=" in hd.get("Set-Cookie", "")
        cookie = hd["Set-Cookie"].split(";")[0]
        assert "s3cr3t" not in cookie  # the cookie carries a derived session value, not the token
        assert request(srv, "/api/runs", {"Cookie": cookie})[0] == 200
        assert request(srv, "/api/runs", {"X-Dashboard-Token": "s3cr3t"})[0] == 200
        assert request(srv, "/api/runs", {"Authorization": "Bearer s3cr3t"})[0] == 200
        assert request(srv, "/files/runs/d1_test/config.json")[0] == 401
        assert request(srv, "/api/bundle/all")[0] == 401
        st, cfg = get_json(srv, "/api/config?token=s3cr3t")
        assert cfg["token_required"] is True
    finally:
        srv.shutdown()
        srv.server_close()


# ----------------------------------------------------------------------------- bundles
def _zip_names(body: bytes):
    zf = zipfile.ZipFile(io.BytesIO(body))
    assert zf.testzip() is None
    return set(zf.namelist()), zf


def test_run_bundle_excludes_weights_by_default(server):
    st, hd, body = request(server, "/api/bundle/run/d1_test")
    assert st == 200 and hd["Content-Type"] == "application/zip" and "attachment" in hd["Content-Disposition"]
    names, zf = _zip_names(body)
    for must in ("runs/d1_test/train_log.jsonl", "runs/d1_test/logs/train.log", "runs/d1_test/status.json",
                 "runs/d1_test/ckpt_0000010/meta.json", "runs/d1_test/last", MANIFEST_NAME):
        assert must in names, must
    assert "runs/d1_test/ckpt_0000010/trainable.pt" not in names and not any(".tmp" in n for n in names)
    m = json.loads(zf.read(MANIFEST_NAME))
    for k in ("created_at", "host", "source_dirs", "files", "git_commit", "python", "packages", "pip_freeze", "nvidia_smi", "skipped"):
        assert k in m, k
    assert any(s["path"].endswith("trainable.pt") for s in m["skipped"])
    assert m["extra"]["kind"] == "run"


def test_run_bundle_with_checkpoints(server):
    st, _, body = request(server, "/api/bundle/run/d1_test?checkpoints=1")
    names, zf = _zip_names(body)
    assert st == 200 and "runs/d1_test/ckpt_0000010/trainable.pt" in names
    assert len(zf.read("runs/d1_test/ckpt_0000010/trainable.pt")) == 1024


def test_eval_and_all_bundles(server):
    _, _, body = request(server, "/api/bundle/eval/eval%2Fd1_test")
    names, _ = _zip_names(body)
    assert "results/eval/d1_test/episodes.jsonl" in names and not any(n.endswith(".mp4") for n in names)
    _, _, body = request(server, "/api/bundle/eval/eval%2Fd1_test?videos=1")
    names, _ = _zip_names(body)
    assert "results/eval/d1_test/videos/seed0_task0_init0_success.mp4" in names
    _, _, body = request(server, "/api/bundle/all?checkpoints=1&videos=1")  # flags ignored for "all"
    names, _ = _zip_names(body)
    assert "runs/d1_test/train_log.jsonl" in names and "results/audit/g0_report_tiny.md" in names
    assert not any(n.endswith((".pt", ".mp4")) for n in names) and "secret.txt" not in " ".join(names)
    assert request(server, "/api/bundle/run/nope")[0] == 404


def test_bundle_cli(tree, tmp_path, capsys):
    out = tmp_path / "b.zip"
    run, ev = tree[0] / "d1_test", tree[1] / "eval" / "d1_test"
    assert bundle_main([str(run), str(ev), "-o", str(out)]) == 0
    names, zf = _zip_names(out.read_bytes())
    assert "d1_test/train_log.jsonl" in names or any(n.endswith("train_log.jsonl") for n in names)
    assert not any(n.endswith(".pt") for n in names)
    m = json.loads(zf.read(MANIFEST_NAME))
    assert m["options"] == {"include_checkpoints": False, "include_videos": False, "max_file_mb": 200}
    assert "wrote" in capsys.readouterr().out
    out2 = tmp_path / "b2.zip"
    assert bundle_main([str(run), "-o", str(out2), "--include-checkpoints"]) == 0
    names2, _ = _zip_names(out2.read_bytes())
    assert any(n.endswith("ckpt_0000010/trainable.pt") for n in names2)
    assert bundle_main([str(tmp_path / "does_not_exist"), "-o", str(tmp_path / "x.zip")]) == 2


def test_bundle_size_cap(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "huge.json").write_bytes(b"{}" + b" " * (2 * 1024 * 1024))
    (d / "big.log").write_bytes(b"x" * (2 * 1024 * 1024))
    buf = io.BytesIO()
    m = write_bundle(buf, [(d, "r")], max_file_mb=1)
    names = {f["path"] for f in m["files"]}
    assert "r/big.log" in names and "r/huge.json" not in names  # logs are never size-capped
    assert any(s["path"] == "r/huge.json" and "larger than" in s["reason"] for s in m["skipped"])
