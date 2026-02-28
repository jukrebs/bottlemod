#!/usr/bin/env python3
"""
Experiment 1 (Companion) — Memory Increase to Eliminate Eviction

Purpose:
  Show that ordering speedup disappears when memory is increased so both
  files fit in cache (no cross-file eviction pressure).

For one file-pair scale (A/B), this script runs the same 4-task workflow
under two memory settings:
  1) eviction memory   (file_size < mem < file_size_a + file_size_b)
  2) no-eviction memory (mem >= file_size_a + file_size_b)

At each memory setting, both orderings are measured:
  - Interleaved: Op1(A) -> Op1(B) -> Op2(A) -> Op2(B)
  - Grouped:     Op1(A) -> Op2(A) -> Op1(B) -> Op2(B)

Expected outcome:
  - Eviction memory: grouped faster than interleaved
  - No-eviction memory: grouped ~= interleaved (ordering speedup ~1.0x)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

_REF_PATH = (
    Path(__file__).resolve().parents[1]
    / "01_cach_aware_ordering"
    / "exp1_reordering.py"
)
_REF_SPEC = importlib.util.spec_from_file_location("exp1_reordering_local", _REF_PATH)
if _REF_SPEC is None or _REF_SPEC.loader is None:
    raise RuntimeError(f"Cannot load module from {_REF_PATH}")
ref = importlib.util.module_from_spec(_REF_SPEC)
sys.modules["exp1_reordering_local"] = ref
_REF_SPEC.loader.exec_module(ref)


def _configure_style() -> None:
    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
    })


def _run_for_memory(
    *,
    video_a: Path,
    video_b: Path,
    out_dir: Path,
    mem_limit: int,
    trials: int,
    drop_caches: bool,
    disk_bw: float,
    mem_bw: float,
) -> dict[str, Any]:
    file_size_a = float(video_a.stat().st_size)
    file_size_b = float(video_b.stat().st_size)

    inter_hits = ref.compute_hit_rates_interleaved(file_size_a, file_size_b, mem_limit)
    grp_hits = ref.compute_hit_rates_grouped(file_size_a, file_size_b, mem_limit)

    inter_files = [file_size_a, file_size_b, file_size_a, file_size_b]
    grp_files = [file_size_a, file_size_a, file_size_b, file_size_b]

    vanilla_a = ref.predict_vanilla(file_size_a, disk_bw)
    vanilla_b = ref.predict_vanilla(file_size_b, disk_bw)
    vanilla_total = 2.0 * vanilla_a + 2.0 * vanilla_b

    inter_ca = [
        ref.predict_cache_aware(inter_files[i], disk_bw, mem_bw, inter_hits[i])[0]
        for i in range(4)
    ]
    grp_ca = [
        ref.predict_cache_aware(grp_files[i], disk_bw, mem_bw, grp_hits[i])[0]
        for i in range(4)
    ]
    inter_ca_total = float(sum(inter_ca))
    grp_ca_total = float(sum(grp_ca))

    inter_trials: list[list[float]] = []
    grp_trials: list[list[float]] = []

    for _ in range(trials):
        t, _ = ref._run_four_task_workflow(
            video_a,
            video_b,
            out_dir,
            ordering="interleaved",
            mem_limit=mem_limit,
            drop_caches=drop_caches,
        )
        inter_trials.append(t)

    for _ in range(trials):
        t, _ = ref._run_four_task_workflow(
            video_a,
            video_b,
            out_dir,
            ordering="grouped",
            mem_limit=mem_limit,
            drop_caches=drop_caches,
        )
        grp_trials.append(t)

    inter_totals = [float(sum(t)) for t in inter_trials]
    grp_totals = [float(sum(t)) for t in grp_trials]

    inter_mean = float(np.mean(inter_totals))
    grp_mean = float(np.mean(grp_totals))
    inter_std = float(np.std(inter_totals))
    grp_std = float(np.std(grp_totals))

    return {
        "mem_limit_bytes": mem_limit,
        "mem_limit_label": ref._format_bytes(mem_limit),
        "hit_rates": {
            "interleaved": inter_hits,
            "grouped": grp_hits,
        },
        "ca": {
            "interleaved_total_s": inter_ca_total,
            "grouped_total_s": grp_ca_total,
            "speedup_x": (inter_ca_total / grp_ca_total) if grp_ca_total > 0 else None,
        },
        "vanilla": {
            "interleaved_total_s": vanilla_total,
            "grouped_total_s": vanilla_total,
            "speedup_x": 1.0,
        },
        "measured": {
            "interleaved_trials": inter_trials,
            "grouped_trials": grp_trials,
            "interleaved_mean_s": inter_mean,
            "grouped_mean_s": grp_mean,
            "interleaved_std_s": inter_std,
            "grouped_std_s": grp_std,
            "speedup_x": (inter_mean / grp_mean) if grp_mean > 0 else None,
        },
    }


def _plot(results: dict[str, Any], out_path: Path) -> None:
    _configure_style()

    scenarios = ["eviction", "no_eviction"]
    labels = [
        f"Eviction\n({results['eviction']['mem_limit_label']})",
        f"No eviction\n({results['no_eviction']['mem_limit_label']})",
    ]
    x = np.arange(len(scenarios))
    width = 0.22

    meas_inter = [results[s]["measured"]["interleaved_mean_s"] for s in scenarios]
    meas_grp = [results[s]["measured"]["grouped_mean_s"] for s in scenarios]
    meas_inter_err = [results[s]["measured"]["interleaved_std_s"] for s in scenarios]
    meas_grp_err = [results[s]["measured"]["grouped_std_s"] for s in scenarios]

    ca_inter = [results[s]["ca"]["interleaved_total_s"] for s in scenarios]
    ca_grp = [results[s]["ca"]["grouped_total_s"] for s in scenarios]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    ax1.bar(x - 1.5 * width, meas_inter, width, yerr=meas_inter_err, capsize=4,
            color="#4C78A8", alpha=0.85, label="Measured interleaved")
    ax1.bar(x - 0.5 * width, meas_grp, width, yerr=meas_grp_err, capsize=4,
            color="#F58518", alpha=0.85, label="Measured grouped")
    ax1.bar(x + 0.5 * width, ca_inter, width,
            color="#4C78A8", alpha=0.35, label="CA interleaved")
    ax1.bar(x + 1.5 * width, ca_grp, width,
            color="#F58518", alpha=0.35, label="CA grouped")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Total runtime (s)")
    ax1.set_title("Ordering effect vs memory regime")
    ax1.legend(frameon=True, fontsize=8, loc="upper right")

    meas_speedups = [results[s]["measured"]["speedup_x"] for s in scenarios]
    ca_speedups = [results[s]["ca"]["speedup_x"] for s in scenarios]

    ax2.plot(x, meas_speedups, marker="o", linewidth=2, color="black", label="Measured")
    ax2.plot(x, ca_speedups, marker="o", linewidth=2, color="#F58518", label="BottleMod-CA")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="No ordering speedup")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Ordering speedup (interleaved / grouped)")
    ax2.set_ylim(0.95, max(1.2, max(meas_speedups + ca_speedups) * 1.05))
    ax2.set_title("Speedup collapses when eviction disappears")
    ax2.legend(frameon=True, fontsize=8, loc="upper right")

    fig.suptitle(results["meta"]["title"], fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Companion experiment: increase memory to remove eviction and ordering speedup"
    )
    ap.add_argument("--video-a", required=True, help="Path to file A")
    ap.add_argument("--video-b", required=True, help="Path to file B")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--mem-evict", required=True, help="Memory limit where eviction exists")
    ap.add_argument("--mem-no-evict", required=True, help="Memory limit where both files fit")
    ap.add_argument("--trials", type=int, default=3, help="Trials per ordering per memory")
    ap.add_argument("--drop-caches", action="store_true", help="Drop caches before each workflow")
    ap.add_argument("--title", default="Memory increase removes ordering speedup", help="Plot title")
    args = ap.parse_args()

    video_a = Path(args.video_a).resolve()
    video_b = Path(args.video_b).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_a.exists():
        raise FileNotFoundError(f"Video A not found: {video_a}")
    if not video_b.exists():
        raise FileNotFoundError(f"Video B not found: {video_b}")

    mem_evict = ref._parse_size_bytes(args.mem_evict)
    mem_no_evict = ref._parse_size_bytes(args.mem_no_evict)

    disk_bw, mem_bw, _ = ref.calibrate(
        video_a,
        out_dir,
        drop_caches=args.drop_caches,
        mem_limit=mem_evict,
    )

    evict = _run_for_memory(
        video_a=video_a,
        video_b=video_b,
        out_dir=out_dir,
        mem_limit=mem_evict,
        trials=args.trials,
        drop_caches=args.drop_caches,
        disk_bw=disk_bw,
        mem_bw=mem_bw,
    )

    no_evict = _run_for_memory(
        video_a=video_a,
        video_b=video_b,
        out_dir=out_dir,
        mem_limit=mem_no_evict,
        trials=args.trials,
        drop_caches=args.drop_caches,
        disk_bw=disk_bw,
        mem_bw=mem_bw,
    )

    results: dict[str, Any] = {
        "schema_version": 1,
        "meta": {
            "host": platform.node(),
            "platform": platform.platform(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "title": args.title,
            "video_a": str(video_a),
            "video_b": str(video_b),
            "file_size_a_bytes": video_a.stat().st_size,
            "file_size_b_bytes": video_b.stat().st_size,
            "trials": args.trials,
            "drop_caches": args.drop_caches,
            "disk_bw_bytes_s": disk_bw,
            "mem_bw_bytes_s": mem_bw,
        },
        "eviction": evict,
        "no_eviction": no_evict,
    }

    out_json = out_dir / "exp1_memory_increase_no_eviction_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    fig_path = out_dir / "fig_memory_increase_no_eviction.png"
    _plot(results, fig_path)

    print("Summary")
    for key, label in [("eviction", "Eviction"), ("no_eviction", "No eviction")]:
        m = results[key]["measured"]
        ca = results[key]["ca"]
        print(
            f"  {label:11s} | Measured inter/group: "
            f"{m['interleaved_mean_s']:.2f}s / {m['grouped_mean_s']:.2f}s "
            f"(speedup={m['speedup_x']:.3f}x) | "
            f"CA speedup={ca['speedup_x']:.3f}x"
        )

    print(f"Results written to: {out_json}")
    print(f"Figure written to:  {fig_path}")


if __name__ == "__main__":
    main()
