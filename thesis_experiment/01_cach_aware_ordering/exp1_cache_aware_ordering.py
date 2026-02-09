#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def _parse_size_bytes(s: str) -> int:
    s = s.strip()
    if s.isdigit():
        return int(s)
    suffixes: dict[str, int] = {
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
    }
    for suf, mul in suffixes.items():
        if s.lower().endswith(suf):
            num = s[: -len(suf)].strip()
            return int(float(num) * mul)
    raise ValueError(f"Unsupported size string: {s!r}")


def _format_bytes(n: int) -> str:
    if n % (1024**3) == 0:
        return f"{n // (1024**3)}GiB"
    if n % (1024**2) == 0:
        return f"{n // (1024**2)}MiB"
    return f"{n}B"


def _run(
    cmd: list[str], *, check: bool = True, capture: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
    )


def _sudo_drop_caches() -> None:
    # Strong cold-start: global side effects, but very repeatable.
    _run(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], check=True)


def _ensure_file(path: Path, size_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == size_bytes:
        return
    # Allocate real blocks (avoid sparse files).
    _run(["fallocate", "-l", str(size_bytes), str(path)], check=True)


def _fio_sequential_read(
    file_path: Path, size_bytes: int, out_json: Path
) -> dict[str, Any]:
    # Buffered sequential read, cache-preserving (invalidate=0).
    cmd = [
        "fio",
        "--name=seqread",
        f"--filename={file_path}",
        "--rw=read",
        "--bs=1m",
        "--iodepth=32",
        "--ioengine=libaio",
        "--direct=0",
        "--invalidate=0",
        "--fadvise_hint=0",
        f"--size={size_bytes}",
        "--numjobs=1",
        "--group_reporting",
        "--output-format=json",
        f"--output={out_json}",
    ]
    _run(cmd, check=True)
    return json.loads(out_json.read_text(encoding="utf-8"))


@dataclass
class StepMeasurement:
    name: str
    runtime_s: float
    bw_bytes_s: float


def _aggregate_fio(payload: dict[str, Any]) -> StepMeasurement:
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("fio JSON missing jobs[]")
    job = jobs[0]
    if not isinstance(job, dict):
        raise ValueError("fio JSON invalid job")
    read = job.get("read")
    if not isinstance(read, dict):
        raise ValueError("fio JSON missing job.read")
    runtime_ms = float(read.get("runtime", 0.0))
    bw_bytes_s = float(read.get("bw_bytes", 0.0))
    return StepMeasurement(
        name=str(job.get("jobname", "unknown")),
        runtime_s=runtime_ms / 1000.0,
        bw_bytes_s=bw_bytes_s,
    )


def _write_jobfile(
    jobfile: Path,
    *,
    file_a: Path,
    file_b: Path,
    a_size: int,
    b_size: int,
    ordering: str,
) -> None:
    # Enforce sequential ordering of jobs via stonewall.
    # Keep buffered IO and preserve cache between jobs.
    if ordering not in {"ABA", "AAB"}:
        raise ValueError("ordering must be ABA or AAB")
    steps: list[tuple[str, Path, int]]
    if ordering == "ABA":
        steps = [("A1", file_a, a_size), ("B", file_b, b_size), ("A2", file_a, a_size)]
    else:
        steps = [("A1", file_a, a_size), ("A2", file_a, a_size), ("B", file_b, b_size)]

    lines: list[str] = []
    lines += [
        "[global]",
        "rw=read",
        "bs=1m",
        "iodepth=32",
        "ioengine=libaio",
        "direct=0",
        "invalidate=0",
        "fadvise_hint=0",
        "numjobs=1",
        "group_reporting=1",
        "",
    ]

    for i, (name, fp, sz) in enumerate(steps):
        lines += [
            f"[{name}]",
            f"filename={fp}",
            f"size={sz}",
        ]
        if i > 0:
            lines.append("stonewall")
        lines.append("")

    jobfile.write_text("\n".join(lines), encoding="utf-8")


def _run_workflow_once(
    *,
    out_dir: Path,
    ordering: str,
    file_a: Path,
    file_b: Path,
    a_size: int,
    b_size: int,
    trial_idx: int,
    drop_caches: bool,
) -> dict[str, StepMeasurement]:
    if drop_caches:
        _sudo_drop_caches()

    jobfile = out_dir / f"job_{ordering}_trial{trial_idx}.fio"
    out_json = out_dir / f"run_{ordering}_trial{trial_idx}.json"
    _write_jobfile(
        jobfile,
        file_a=file_a,
        file_b=file_b,
        a_size=a_size,
        b_size=b_size,
        ordering=ordering,
    )

    _run(
        ["fio", "--output-format=json", f"--output={out_json}", str(jobfile)],
        check=True,
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("fio output missing jobs")

    out: dict[str, StepMeasurement] = {}
    for job in jobs:
        if not isinstance(job, dict):
            continue
        name = str(job.get("jobname", ""))
        # Re-wrap to match _aggregate_fio interface.
        out[name] = _aggregate_fio({"jobs": [job]})
    return out


def _configure_matplotlib_paper_style() -> None:
    matplotlib.rcParams.update(
        {
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
        }
    )


def _plot_fig6(
    *,
    out_path: Path,
    x_pct: list[float],
    predicted_s: list[float],
    mean_s: list[float],
    min_s: list[float],
    max_s: list[float],
    title: str,
) -> None:
    # Mimic paper Figure 6: orange prediction line, black min/max bars for measured avg.
    _configure_matplotlib_paper_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))

    ax.plot(
        x_pct,
        predicted_s,
        color="#F58518",
        linewidth=2.0,
        label="BottleMod-SH prediction",
    )

    yerr = np.array(
        [
            np.array(mean_s) - np.array(min_s),
            np.array(max_s) - np.array(mean_s),
        ]
    )
    ax.errorbar(
        x_pct,
        mean_s,
        yerr=yerr,
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
        label="Measured (min/max)",
    )

    ax.set_xlabel("B size / cache budget (%)")
    ax.set_ylabel("Total workflow time (s)")
    ax.set_title(title)
    ax.legend(frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_fig7(
    *,
    out_path: Path,
    baseline_progress_xy: list[tuple[float, float]],
    fix_progress_xy: list[tuple[float, float]],
    baseline_segments: list[tuple[float, float, str]],
    fix_segments: list[tuple[float, float, str]],
    baseline_total_s: float,
    fix_total_s: float,
    baseline_disk_rate: list[tuple[float, float, float, float]],
    fix_disk_rate: list[tuple[float, float, float, float]],
) -> None:
    """Paper Figure 7 style: 2x2 (baseline vs fix columns).

    baseline_segments/fix_segments: list of (t0, t1, label) with label in {A_disk, A_cache, B_disk}
    baseline_disk_rate/fix_disk_rate: list of (t0, t1, A_rate, B_rate) in bytes/s
    """
    _configure_matplotlib_paper_style()

    colors = {
        "A_disk": "#F58518",  # orange
        "A_cache": "#4C78A8",  # blue
        "B_disk": "#54A24B",  # green
    }

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 4.8), sharex="col")

    def plot_progress(
        ax: Axes,
        progress_xy: list[tuple[float, float]],
        segments: list[tuple[float, float, str]],
        y_on_right: bool,
    ) -> None:
        for t0, t1, lbl in segments:
            ax.axvspan(t0, t1, color=colors[lbl], alpha=0.25, linewidth=0)
        xs = [t for t, _ in progress_xy]
        ys = [p for _, p in progress_xy]
        ax.plot(xs, ys, color="#4C78A8", linewidth=1.8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("progress [%]")
        if y_on_right:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
        else:
            ax.yaxis.set_label_position("left")
            ax.yaxis.tick_left()

    def plot_rates(
        ax: Axes,
        total_s: float,
        rate_segments: list[tuple[float, float, float, float]],
        y_on_right: bool,
    ) -> None:
        # Build piecewise-constant lines.
        xs_a: list[float] = [0.0]
        ys_a: list[float] = [0.0]
        xs_b: list[float] = [0.0]
        ys_b: list[float] = [0.0]
        for t0, t1, a_rate, b_rate in rate_segments:
            xs_a += [t0, t1]
            ys_a += [a_rate / 1e6, a_rate / 1e6]
            xs_b += [t0, t1]
            ys_b += [b_rate / 1e6, b_rate / 1e6]
        xs_a.append(total_s)
        ys_a.append(0.0)
        xs_b.append(total_s)
        ys_b.append(0.0)
        ax.plot(xs_a, ys_a, color="#F58518", linewidth=2.0, label="A disk")
        ax.plot(
            xs_b, ys_b, color="#54A24B", linewidth=2.0, linestyle="--", label="B disk"
        )
        ax.set_ylabel("data rate [MB/second]")
        ax.set_xlabel("time [seconds]")
        if y_on_right:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
        else:
            ax.yaxis.set_label_position("left")
            ax.yaxis.tick_left()

    # Top row: progress + bottlenecks
    plot_progress(axes[0][0], baseline_progress_xy, baseline_segments, y_on_right=False)
    plot_progress(axes[0][1], fix_progress_xy, fix_segments, y_on_right=True)
    axes[0][0].set_title("Baseline (A→B→A)")
    axes[0][1].set_title("Fix (A→A→B)")

    # Legend like paper: "Limited by"
    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor=colors["A_disk"], alpha=0.25, label="A from disk"),
        Patch(facecolor=colors["A_cache"], alpha=0.25, label="A from cache"),
        Patch(facecolor=colors["B_disk"], alpha=0.25, label="B from disk"),
    ]
    axes[0][0].legend(
        handles=legend_elems, title="Limited by:", frameon=True, loc="upper left"
    )

    # Bottom row: disk data rates
    plot_rates(axes[1][0], baseline_total_s, baseline_disk_rate, y_on_right=False)
    plot_rates(axes[1][1], fix_total_s, fix_disk_rate, y_on_right=True)
    axes[1][1].legend(frameon=True, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-bytes", default="4G")
    ap.add_argument("--a-bytes", default="2G")
    ap.add_argument(
        "--b-bytes-sweep",
        default="0,1G,2G,3G,4G,6G,8G",
        help="Comma-separated sizes for B sweep (e.g., 0,1G,2G)",
    )
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--drop-caches", action="store_true")
    ap.add_argument("--cache-effective-fraction", type=float, default=0.85)
    ap.add_argument("--data-dir", default="/var/tmp/bottlemod_thesis_exp1")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    cache_bytes = _parse_size_bytes(args.cache_bytes)
    a_bytes = _parse_size_bytes(args.a_bytes)
    b_sweep = [_parse_size_bytes(x) for x in args.b_bytes_sweep.split(",") if x.strip()]

    file_a = data_dir / f"fileA_{a_bytes}.bin"
    _ensure_file(file_a, a_bytes)

    # We'll re-create B per size to avoid fragmentation surprises.
    # (Re-using a max-sized file would be fine too.)

    # Calibration: disk_bw (cold) and mem_bw (warm)
    calib_dir = out_dir / "calibration"
    calib_dir.mkdir(exist_ok=True)
    if args.drop_caches:
        _sudo_drop_caches()
    cold_payload = _fio_sequential_read(file_a, a_bytes, calib_dir / "A_cold.json")
    cold = _aggregate_fio(cold_payload)
    warm_payload = _fio_sequential_read(file_a, a_bytes, calib_dir / "A_warm.json")
    warm = _aggregate_fio(warm_payload)
    disk_bw = a_bytes / max(cold.runtime_s, 1e-9)
    mem_bw = a_bytes / max(warm.runtime_s, 1e-9)

    c_eff = cache_bytes * float(args.cache_effective_fraction)

    results: dict[str, Any] = {
        "schema_version": 1,
        "meta": {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "cache_bytes": cache_bytes,
            "cache_effective_fraction": float(args.cache_effective_fraction),
            "cache_effective_bytes": c_eff,
            "a_bytes": a_bytes,
            "b_sweep_bytes": b_sweep,
            "trials": int(args.trials),
            "drop_caches": bool(args.drop_caches),
            "disk_bw_bytes_s": disk_bw,
            "mem_bw_bytes_s": mem_bw,
        },
        "series": {},
    }

    series: dict[str, Any] = {}

    for ordering in ("ABA", "AAB"):
        series[ordering] = {"points": []}

    for b_bytes in b_sweep:
        file_b = data_dir / f"fileB_{b_bytes}.bin"
        if b_bytes > 0:
            _ensure_file(file_b, b_bytes)
        else:
            # fio expects a filename; reuse A but size 0 makes job a no-op.
            file_b = file_a

        x_pct = 100.0 * (b_bytes / cache_bytes)

        # Simple reuse heuristic for baseline: fraction of A retained after reading B.
        # If B fits in remaining cache, A stays (hit=1). If B exceeds cache, A is evicted (hit=0).
        # Linear interpolation for partial eviction.
        # Clamp in [0,1].
        if b_bytes <= max(0.0, c_eff - a_bytes):
            a2_hit = 1.0
        else:
            a2_hit = max(0.0, min(1.0, (c_eff - float(b_bytes)) / float(a_bytes)))

        pred_a_cold = a_bytes / disk_bw
        pred_b_cold = b_bytes / disk_bw if b_bytes > 0 else 0.0
        pred_a_warm = a_bytes / mem_bw
        pred_a2_mix = (a2_hit * a_bytes) / mem_bw + ((1.0 - a2_hit) * a_bytes) / disk_bw

        pred_aba = pred_a_cold + pred_b_cold + pred_a2_mix
        pred_aab = pred_a_cold + pred_a_warm + pred_b_cold

        point: dict[str, Any] = {
            "b_bytes": b_bytes,
            "x_pct": x_pct,
            "predicted_total_s": {"ABA": pred_aba, "AAB": pred_aab},
            "predicted_a2_hit_rate": a2_hit,
            "measured": {"ABA": [], "AAB": []},
        }

        for ordering in ("ABA", "AAB"):
            for trial in range(1, int(args.trials) + 1):
                meas = _run_workflow_once(
                    out_dir=out_dir,
                    ordering=ordering,
                    file_a=file_a,
                    file_b=file_b,
                    a_size=a_bytes,
                    b_size=b_bytes,
                    trial_idx=trial,
                    drop_caches=bool(args.drop_caches),
                )
                # normalize missing jobs in size=0 case
                a1 = meas.get("A1")
                a2 = meas.get("A2")
                b = meas.get("B")
                if a1 is None or a2 is None or b is None:
                    raise RuntimeError(
                        f"Missing expected fio jobs in output: {sorted(meas.keys())}"
                    )
                total = a1.runtime_s + a2.runtime_s + b.runtime_s
                point["measured"][ordering].append(
                    {
                        "trial": trial,
                        "A1_s": a1.runtime_s,
                        "A2_s": a2.runtime_s,
                        "B_s": b.runtime_s,
                        "total_s": total,
                        "A1_bw": a1.bw_bytes_s,
                        "A2_bw": a2.bw_bytes_s,
                        "B_bw": b.bw_bytes_s,
                    }
                )

        series.setdefault("points", []).append(point)

    # Reformat series to per-ordering, keep points list identical order
    points = series["points"]
    results["series"] = {"points": points}

    out_json = out_dir / "exp1_cache_aware_ordering_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Build Figure-6 style plots for baseline and fix
    x = [p["x_pct"] for p in points]
    for ordering, title in [("ABA", "Baseline (A→B→A)"), ("AAB", "Fix (A→A→B)")]:
        means: list[float] = []
        mins: list[float] = []
        maxs: list[float] = []
        preds: list[float] = []
        for p in points:
            totals = [m["total_s"] for m in p["measured"][ordering]]
            means.append(float(np.mean(totals)))
            mins.append(float(np.min(totals)))
            maxs.append(float(np.max(totals)))
            preds.append(float(p["predicted_total_s"][ordering]))
        _plot_fig6(
            out_path=out_dir / f"fig6_exp1_{ordering}.png",
            x_pct=x,
            predicted_s=preds,
            mean_s=means,
            min_s=mins,
            max_s=maxs,
            title=title,
        )

    # Figure-7 style: choose a representative "large B" point (max B)
    p_big = max(points, key=lambda p: int(p["b_bytes"]))
    b_big = int(p_big["b_bytes"])
    # Use predictions to construct segments.
    tA_cold = a_bytes / disk_bw
    tB = b_big / disk_bw if b_big > 0 else 0.0
    # baseline A2 uses mix; show as disk if hit<0.5 else cache
    hit = float(p_big["predicted_a2_hit_rate"])
    tA2_base = float(p_big["predicted_total_s"]["ABA"]) - (tA_cold + tB)
    tA2_fix = a_bytes / mem_bw

    base_segments: list[tuple[float, float, str]] = []
    t = 0.0
    base_segments.append((t, t + tA_cold, "A_disk"))
    t += tA_cold
    if tB > 0:
        base_segments.append((t, t + tB, "B_disk"))
        t += tB
    base_segments.append((t, t + tA2_base, "A_disk" if hit < 0.95 else "A_cache"))
    base_total = t + tA2_base

    fix_segments: list[tuple[float, float, str]] = []
    t = 0.0
    fix_segments.append((t, t + tA_cold, "A_disk"))
    t += tA_cold
    fix_segments.append((t, t + tA2_fix, "A_cache"))
    t += tA2_fix
    if tB > 0:
        fix_segments.append((t, t + tB, "B_disk"))
        t += tB
    fix_total = t

    base_rates: list[tuple[float, float, float, float]] = []
    t = 0.0
    base_rates.append((t, t + tA_cold, disk_bw, 0.0))
    t += tA_cold
    if tB > 0:
        base_rates.append((t, t + tB, 0.0, disk_bw))
        t += tB
    base_rates.append((t, t + tA2_base, disk_bw, 0.0))

    fix_rates: list[tuple[float, float, float, float]] = []
    t = 0.0
    fix_rates.append((t, t + tA_cold, disk_bw, 0.0))
    t += tA_cold
    fix_rates.append((t, t + tA2_fix, 0.0, 0.0))
    t += tA2_fix
    if tB > 0:
        fix_rates.append((t, t + tB, 0.0, disk_bw))

    _plot_fig7(
        out_path=out_dir / "fig7_exp1_baseline_vs_fix.png",
        baseline_progress_xy=[
            (0.0, 0.0),
            (tA_cold, 100.0 * (a_bytes / (2 * a_bytes + b_big))),
            (tA_cold + tB, 100.0 * ((a_bytes + b_big) / (2 * a_bytes + b_big))),
            (base_total, 100.0),
        ],
        fix_progress_xy=[
            (0.0, 0.0),
            (tA_cold, 100.0 * (a_bytes / (2 * a_bytes + b_big))),
            (tA_cold + tA2_fix, 100.0 * ((2 * a_bytes) / (2 * a_bytes + b_big))),
            (fix_total, 100.0),
        ],
        baseline_segments=base_segments,
        fix_segments=fix_segments,
        baseline_total_s=base_total,
        fix_total_s=fix_total,
        baseline_disk_rate=base_rates,
        fix_disk_rate=fix_rates,
    )

    print(f"Wrote results: {out_json}")
    print(
        f"Plots: {out_dir}/fig6_exp1_ABA.png, {out_dir}/fig6_exp1_AAB.png, {out_dir}/fig7_exp1_baseline_vs_fix.png"
    )


if __name__ == "__main__":
    main()
