"""Assemble the Phase-1 results report from evaluation outputs (never invents numbers).

  python -m monet_action.report --eval-dir results/eval --out results/phase1_report.md

Reads ``<eval-dir>/<run>/{summary.json, episodes.jsonl, offline.json, latency.json}``; anything
missing is reported as "not run".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import compare
from .metrics import markdown_table

PRINCIPAL = ["d0_k0", "d1_k10", "d2_lora", "d0_k0_lora"]
CONTROLS = ["d4_query", "d4_query_lora", "d5_mlp", "d1_latents_only", "d1_no_toolresult", "d1_sft_convention", "d1_k5", "d1_k20"]


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def build_report(eval_dir: str) -> str:
    root = Path(eval_dir)
    runs = {p.name: p for p in sorted(root.iterdir()) if (p / "summary.json").exists()} if root.exists() else {}
    S = {k: _load(p / "summary.json") for k, p in runs.items()}
    L = ["# Monet-Action Phase 1 results", "", f"Source: `{root}`. Paused-simulator LIBERO-Spatial task success; "
         "no real-time or physical-safety claim.", ""]
    if not runs:
        return "\n".join(L + ["No evaluation outputs found (not run)."])

    L += ["## 1. Main comparison (matched data, seeds, DiT, init states)", ""]
    main = {k: S[k] for k in PRINCIPAL + CONTROLS if k in S}
    L += [markdown_table(main) if main else "not run", ""]
    present = [str(runs[k]) for k in PRINCIPAL + ["d4_query"] if k in runs]
    if len(present) >= 2:
        L += ["## 2. Paired statistics", "", compare(present), ""]

    L += ["## 3. Per-task success (successes / trials)", ""]
    names = next(iter(S.values())).get("task_names", {})
    cols = [k for k in PRINCIPAL + CONTROLS if k in S]
    if cols:
        L.append("| task | " + " | ".join(cols) + " |")
        L.append("|---|" + "---|" * len(cols))
        tids = sorted({int(t) for k in cols for t in S[k]["per_task"]})
        for t in tids:
            row = [f"{S[k]['per_task'].get(str(t), S[k]['per_task'].get(t, {})).get('successes', '-')}/"
                   f"{S[k]['per_task'].get(str(t), S[k]['per_task'].get(t, {})).get('trials', '-')}" for k in cols]
            L.append(f"| {t} {names.get(str(t), names.get(t, ''))[:60]} | " + " | ".join(row) + " |")
        L.append("")

    L += ["## 4. D3 latent interventions on D2 (perturbation diagnostic, OOD by construction)", ""]
    d3 = {k: S[k] for k in S if k.startswith("d3_")}
    if d3 and "d2_lora" in S:
        base = S["d2_lora"]["success_rate"]
        L += ["| intervention | success | delta vs D2 |", "|---|---|---|"]
        for k, s in d3.items():
            L.append(f"| {k[3:]} | {s['success_rate']:.3f} | {s['success_rate'] - base:+.3f} |")
    else:
        L.append("not run")
    L.append("")

    L += ["## 4b. K sweep (frozen Monet; K=0 is D0)", "", "| K | run | success | 95% CI | monet p50 ms | GPU-s / success |",
          "|---|---|---|---|---|---|"]
    ks = [(0, "d0_k0"), (5, "d1_k5"), (10, "d1_k10"), (20, "d1_k20")]
    if any(k in S for _, k in ks):
        for K, k in ks:
            if k in S:
                s_ = S[k]
                mon = s_.get("latency_ms_per_call", {}).get("monet", {}).get("p50", float("nan"))
                L.append(f"| {K} | {k} | {s_['success_rate']:.3f} | [{s_['success_ci95'][0]:.3f}, {s_['success_ci95'][1]:.3f}] "
                         f"| {mon:.1f} | {s_['gpu_seconds_per_success']:.2f} |")
    else:
        L.append("| not run | | | | | |")
    L += ["", "## 4c. Automatic failure-tag counts (heuristic hints; confirm on videos)", ""]
    tagged = {k: s_.get("failure_tag_counts") for k, s_ in S.items() if s_.get("failure_tag_counts")}
    if tagged:
        L += ["| run | tags |", "|---|---|"] + [f"| {k} | {', '.join(f'{t}: {n}' for t, n in v.items())} |"
                                              for k, v in tagged.items()]
    else:
        L.append("no failures tagged / not run")
    L.append("")

    L += ["## 5. Execution prefix and solver steps", ""]
    extra = {k: S[k] for k in S if k.endswith("_E2") or "_steps" in k}
    L += [markdown_table(extra) if extra else "not run", ""]

    L += ["## 6. Open-loop held-out metrics (fixed noise)", "", "| run | steps | fm_loss | action_mse | first-action mse | gripper acc |",
          "|---|---|---|---|---|---|"]
    any_off = False
    for k, p in runs.items():
        off = _load(p / "offline.json")
        if off:
            any_off = True
            for st, m in off.items():
                L.append(f"| {k} | {st.split('_')[-1]} | {m['fm_loss']:.4f} | {m['action_mse']:.4f} | "
                         f"{m['action_mse_first']:.4f} | {m['gripper_acc']:.3f} |")
    if not any_off:
        L.append("| not run | | | | | |")
    L.append("")

    L += ["## 7. Batch-1 latency (ms, p50 / p95) and peak memory", "",
          "| run | K | Euler steps | monet | action head | total | peak GB |", "|---|---|---|---|---|---|---|"]
    any_lat = False
    for k, p in runs.items():
        lat = _load(p / "latency.json")
        if lat:
            any_lat = True
            for st, prof in lat["profile"].items():
                f = lambda c, prof=prof: f"{prof.get(c, {}).get('p50', float('nan')):.1f} / {prof.get(c, {}).get('p95', float('nan')):.1f}"
                L.append(f"| {k} | {lat['K']} | {st.split('_')[-1]} | {f('monet')} | {f('action_head')} | {f('total')} | "
                         f"{prof.get('peak_mem_gb', float('nan')):.1f} |")
    if not any_lat:
        L.append("| not run | | | | | | |")
    L += ["", "## 8. Cost-quality frontier", "", "| run | success | GPU-s / episode | GPU-s / success |", "|---|---|---|---|"]
    for k, s in sorted(S.items(), key=lambda kv: kv[1]["gpu_seconds_per_episode"]):
        L.append(f"| {k} | {s['success_rate']:.3f} | {s['gpu_seconds_per_episode']:.2f} | {s['gpu_seconds_per_success']:.2f} |")
    L += ["", "Interpretation must follow the pre-registered questions in RESEARCH_MEMO.md (H1-H4)."]
    return "\n".join(L)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="results/eval")
    ap.add_argument("--out", default="results/phase1_report.md")
    a = ap.parse_args(argv)
    text = build_report(a.eval_dir)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
