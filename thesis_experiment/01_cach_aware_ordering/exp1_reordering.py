#!/usr/bin/env python3
"""
Experiment 1 — Cache-Aware Task Ordering with Two Same-Size Files

Demonstrates that task ordering matters when two files A and B share a
constrained page cache.  Four sequential ffmpeg remux operations are
executed under a fixed 5 GB cgroup memory limit in two orderings:

  Interleaved (slow): Remux1(A) -> Remux1(B) -> Remux2(A) -> Remux2(B)
      Each file evicts the other's pages; every operation is cold.

  Grouped (fast):     Remux1(A) -> Remux2(A) -> Remux1(B) -> Remux2(B)
      The second operation on each file benefits from cached pages.

Three models are compared for each ordering:
  1. Real ffmpeg measurement
  2. Vanilla BottleMod prediction (no cache awareness)
  3. BottleMod-CA prediction (cache-aware, models eviction between tasks)

Produces paper-style 2x2 plots (progress + bottleneck bands on top,
resource consumption on bottom) for each ordering, plus a summary bar
chart comparing all three models.

Usage:
    ROOT="$HOME/bm_exp/bottlemod_cache_aware"
    PY="$ROOT/.venv/bin/python"
    PYTHONPATH="$ROOT" "$PY" \\
        "$ROOT/thesis_experiment/01_cach_aware_ordering/exp1_reordering.py" \\
        --video-a /mnt/sata/input.mp4 \\
        --video-b /mnt/sata/input_b.mp4 \\
        --out-dir "/var/tmp/exp1_reordering_$(date +%Y%m%d_%H%M%S)" \\
        --mem-limit 5G --trials 5 --drop-caches
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from bottlemod.func import Func

# ---------------------------------------------------------------------------
# BottleMod-CA imports
# ---------------------------------------------------------------------------
from bottlemod.ppoly import PPoly
from bottlemod.storage_hierarchy import (
    LRUEvictionModel,
    LogicalAccessProfile,
    StorageHierarchyTask,
    StorageTier,
    TierMapping,
    get_bottleneck_label,
)
from bottlemod.task import TaskExecution

# ---------------------------------------------------------------------------
# Vanilla BottleMod imports
# ---------------------------------------------------------------------------
_VANILLA_DIR = str(Path(__file__).resolve().parents[2] / "bottlemod_vanilla")
if _VANILLA_DIR not in sys.path:
    sys.path.insert(0, _VANILLA_DIR)

_vanilla_task_mod = importlib.import_module("task")
_vanilla_func_mod = importlib.import_module("func")
_vanilla_ppoly_mod = importlib.import_module("ppoly")
VanillaTask = _vanilla_task_mod.Task
VanillaTaskExecution = _vanilla_task_mod.TaskExecution
VanillaFunc = _vanilla_func_mod.Func
VanillaPPoly = _vanilla_ppoly_mod.PPoly

sys.path.remove(_VANILLA_DIR)


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_size_bytes(s: str) -> int:
    s = s.strip()
    if s.isdigit():
        return int(s)
    suffixes: dict[str, int] = {
        "k": 1024, "kb": 1024,
        "m": 1024**2, "mb": 1024**2,
        "g": 1024**3, "gb": 1024**3,
        "t": 1024**4, "tb": 1024**4,
    }
    for suf, mul in suffixes.items():
        if s.lower().endswith(suf):
            num = s[: -len(suf)].strip()
            return int(float(num) * mul)
    raise ValueError(f"Unsupported size string: {s!r}")


def _format_bytes(n: int) -> str:
    if n >= 1024**3 and n % (1024**3) == 0:
        return f"{n // (1024**3)}G"
    if n >= 1024**2 and n % (1024**2) == 0:
        return f"{n // (1024**2)}M"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return f"{n}B"


def _format_bytes_human(n: int) -> str:
    if n >= 1024**3:
        return f"{n / (1024**3):.1f} GB"
    if n >= 1024**2:
        return f"{n / (1024**2):.0f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


def _run(
    cmd: list[str], *, check: bool = True, capture: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def _sudo_drop_caches() -> None:
    _run(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], check=True)


# ===========================================================================
# FFmpeg runner
# ===========================================================================


def _run_ffmpeg_remux(
    video_in: Path,
    video_out: Path,
    *,
    mem_limit: int | None = None,
    drop_caches: bool = False,
) -> float:
    """Run ffmpeg remux and return wall-clock time in seconds."""
    if drop_caches:
        _sudo_drop_caches()

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-i", str(video_in), "-c", "copy", str(video_out),
    ]

    if mem_limit is not None:
        cmd = [
            "sudo", "systemd-run", "--wait", "--collect", "--quiet",
            f"--property=MemoryMax={mem_limit}",
            "--property=MemorySwapMax=0", "--",
        ] + ffmpeg_cmd
    else:
        cmd = ffmpeg_cmd

    t0 = time.monotonic()
    _run(cmd, check=True, capture=True)
    t1 = time.monotonic()

    if video_out.exists():
        video_out.unlink()

    return t1 - t0


# ===========================================================================
# Calibration
# ===========================================================================


def calibrate(
    video: Path,
    out_dir: Path,
    *,
    drop_caches: bool,
    mem_limit: int,
) -> Tuple[float, float, int]:
    """Measure cold (disk) and warm (cache) bandwidth.

    Cold run uses the given *mem_limit* cgroup.
    Warm run uses no cgroup (all RAM available) after pre-loading.

    Returns ``(disk_bw_bytes_s, mem_bw_bytes_s, file_size_bytes)``.
    """
    file_size = video.stat().st_size
    tmp_out = out_dir / "calib_output.mp4"

    print(f"Calibrating: file_size = {file_size / 1e9:.2f} GB")

    # Cold: cgroup-constrained, caches dropped
    cold_time = _run_ffmpeg_remux(
        video, tmp_out, mem_limit=mem_limit, drop_caches=drop_caches,
    )
    disk_bw = file_size / max(cold_time, 1e-9)
    print(
        f"  Cold run (cgroup {_format_bytes(mem_limit)}): "
        f"{cold_time:.2f}s  ->  disk_bw = {disk_bw / 1e6:.1f} MB/s"
    )

    # Warm: pre-load into page cache (no cgroup), then measure
    _run_ffmpeg_remux(video, tmp_out)  # pre-load
    warm_time = _run_ffmpeg_remux(video, tmp_out)  # measure from cache
    mem_bw = file_size / max(warm_time, 1e-9)
    print(
        f"  Warm run (no cgroup): "
        f"{warm_time:.2f}s  ->  mem_bw  = {mem_bw / 1e6:.1f} MB/s"
    )

    return disk_bw, mem_bw, file_size


# ===========================================================================
# Modeling
# ===========================================================================


def predict_vanilla(file_size: float, disk_bw: float) -> float:
    """Vanilla BottleMod: constant file_size / disk_bw regardless of cache."""
    max_progress = float(file_size)
    T_max = max_progress / disk_bw

    out_cpu = [VanillaPPoly([0, max_progress], [[1e-9]])]
    out_data = [VanillaFunc([0, max_progress], [[1, 0]])]
    in_cpu = [VanillaPPoly([0, T_max], [[1e12]])]
    in_data = [VanillaFunc([0, T_max], [[disk_bw, 0]])]

    task = VanillaTask(out_cpu, out_data)
    result, _ = VanillaTaskExecution(task, in_cpu, in_data).get_result()
    return float(result.x[-1])


def _make_sh_task(
    file_size: float, disk_bw: float, mem_bw: float, hit_rate: float,
) -> StorageHierarchyTask:
    max_progress = 1.0
    T_max = file_size / disk_bw * 2.0

    return StorageHierarchyTask(
        access_profiles=[
            LogicalAccessProfile.sequential_read("video", file_size, max_progress)
        ],
        tier_mappings=[
            TierMapping.constant_hit_rate(
                "video",
                cache_tier=0,
                backing_tier=1,
                hit_rate=hit_rate,
                progress_range=(0, max_progress),
            )
        ],
        tiers=[
            StorageTier(
                name="PageCache", tier_index=0,
                I_bw_read=PPoly([0, T_max], [[mem_bw]]),
            ),
            StorageTier(
                name="Disk", tier_index=1,
                I_bw_read=PPoly([0, T_max], [[disk_bw]]),
            ),
        ],
        cpu_funcs=[PPoly([0, max_progress], [[1e-9]])],
        data_funcs=[Func([0, max_progress], [[1, 0]])],
    )


def predict_cache_aware(
    file_size: float, disk_bw: float, mem_bw: float, hit_rate: float,
) -> Tuple[float, Any, Any, StorageHierarchyTask]:
    max_progress = 1.0
    T_max = file_size / disk_bw * 2.0

    sh_task = _make_sh_task(file_size, disk_bw, mem_bw, hit_rate)

    bm_task = sh_task.to_task()
    storage_inputs = sh_task.get_storage_input_funcs()

    in_cpu = [PPoly([0, T_max], [[1e12]])]
    all_in_cpu = in_cpu + storage_inputs
    in_data = [Func([0, T_max], [[max_progress]])]

    execution = TaskExecution(bm_task, all_in_cpu, in_data)
    progress, bottlenecks = execution.get_result()
    return float(progress.x[-1]), progress, bottlenecks, sh_task


# ===========================================================================
# Cache eviction model for sequential task chains
# ===========================================================================


def compute_hit_rates_interleaved(
    file_size_a: float, file_size_b: float, mem_limit: int,
) -> list[float]:
    """Hit rates for interleaved ordering: Op1(A), Op1(B), Op2(A), Op2(B)."""
    model = LRUEvictionModel(cache_capacity_bytes=float(mem_limit))
    return model.compute_hit_rates([
        ("A", file_size_a),
        ("B", file_size_b),
        ("A", file_size_a),
        ("B", file_size_b),
    ])


def compute_hit_rates_grouped(
    file_size_a: float, file_size_b: float, mem_limit: int,
) -> list[float]:
    """Hit rates for grouped ordering: Op1(A), Op2(A), Op1(B), Op2(B)."""
    model = LRUEvictionModel(cache_capacity_bytes=float(mem_limit))
    return model.compute_hit_rates([
        ("A", file_size_a),
        ("A", file_size_a),
        ("B", file_size_b),
        ("B", file_size_b),
    ])


# ===========================================================================
# Workflow runners (measurement)
# ===========================================================================


def _run_four_task_workflow(
    video_a: Path,
    video_b: Path,
    out_dir: Path,
    *,
    ordering: str,
    mem_limit: int,
    drop_caches: bool,
) -> Tuple[list[float], float]:
    """Run 4 sequential ffmpeg remux tasks inside a **single persistent cgroup**.

    *ordering*: "interleaved" or "grouped".

    All four tasks share a single cgroup so that page-cache pressure is
    realistic: pages cached by task N remain charged to the cgroup when
    task N+1 starts.  A new ``systemd-run --scope`` is created once for
    the whole workflow; a small bash script inside it runs each ffmpeg
    sequentially and records per-task wall-clock times to a temp file.

    IMPORTANT: Only drop caches before the first task.  Cache state carries
    over between tasks — that's the whole point of this experiment.
    """
    tmp_out = out_dir / "tmp_workflow_output.mp4"
    timing_file = out_dir / "tmp_task_times.txt"

    if ordering == "interleaved":
        files = [video_a, video_b, video_a, video_b]
    elif ordering == "grouped":
        files = [video_a, video_a, video_b, video_b]
    else:
        raise ValueError(f"Unknown ordering: {ordering!r}")

    if drop_caches:
        _sudo_drop_caches()

    # Build a bash script that runs all 4 ffmpeg tasks sequentially inside
    # the same process (and therefore the same cgroup), recording per-task
    # wall-clock times.
    script_lines = [
        "#!/bin/bash",
        "set -e",
        f"TIMING_FILE={str(timing_file)}",
        f"TMP_OUT={str(tmp_out)}",
        "> $TIMING_FILE",
    ]
    for i, f in enumerate(files):
        script_lines += [
            f"T0=$(date +%s%N)",
            f'ffmpeg -y -nostdin -loglevel error -i {str(f)} -c copy "$TMP_OUT"',
            f"T1=$(date +%s%N)",
            f'echo "$(( T1 - T0 ))" >> "$TIMING_FILE"',
            f'rm -f "$TMP_OUT"',
        ]

    script = "\n".join(script_lines) + "\n"

    cmd = [
        "sudo", "systemd-run", "--collect", "--quiet",
        "--scope",
        f"--property=MemoryMax={mem_limit}",
        "--property=MemorySwapMax=0",
        "--", "bash", "-c", script,
    ]

    t_total_0 = time.monotonic()
    _run(cmd, check=True, capture=True)
    t_total_1 = time.monotonic()

    # Parse per-task nanosecond deltas from the timing file
    times: list[float] = []
    with open(timing_file) as fh:
        for line in fh:
            ns = int(line.strip())
            times.append(ns / 1e9)

    timing_file.unlink(missing_ok=True)

    if len(times) != 4:
        raise RuntimeError(
            f"Expected 4 task times, got {len(times)}: {times}"
        )

    return times, sum(times)


# ===========================================================================
# Plotting
# ===========================================================================


COLOR_MAP = {
    "disk_bw": "#F58518",   # orange
    "memory_bw": "#4C78A8", # blue
    "cpu": "#54A24B",       # green
    "data": "#999999",      # gray
}


def _configure_matplotlib_paper_style() -> None:
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


def _bottleneck_category(bn_index: int, sh_task: StorageHierarchyTask) -> str:
    label = get_bottleneck_label(bn_index, sh_task).lower()
    if "pagecache" in label or "mem" in label or "cache" in label:
        return "memory_bw"
    elif "disk" in label:
        return "disk_bw"
    elif "cpu" in label:
        return "cpu"
    return "data"


def _plot_progress_panel(
    ax: Axes,
    progress: Any,
    bottlenecks: list[int],
    sh_task: StorageHierarchyTask,
    title: str,
) -> None:
    """Plot progress (%) with bottleneck-coloured background bands."""
    T_end = float(progress.x[-1])
    for i, seg_x in enumerate(progress.x[:-1]):
        seg_end = progress.x[i + 1]
        if i < len(bottlenecks):
            cat = _bottleneck_category(bottlenecks[i], sh_task)
            color = COLOR_MAP.get(cat, "#999999")
        else:
            color = "#999999"
        ax.axvspan(float(seg_x), float(seg_end), color=color, alpha=0.25, linewidth=0)

    xs = np.linspace(float(progress.x[0]), T_end, 500)
    max_val = float(progress(T_end))
    ys = [float(progress(xv)) / max_val * 100.0 if max_val > 0 else 0.0 for xv in xs]
    ax.plot(xs, ys, color="#4C78A8", linewidth=1.8)
    ax.set_ylim(0, 105)
    ax.set_title(title, fontsize=9)


def _to_float(value: object) -> float:
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        return float(value.reshape(-1)[0])
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_float(item())
        except Exception:
            pass
    return 0.0


def _plot_resource_panel(
    ax: Axes,
    progress: Any,
    sh_task: StorageHierarchyTask,
    disk_bw: float,
    mem_bw: float,
) -> None:
    """Plot bandwidth usage per tier over time (paper-style bottom row)."""
    T_end = float(progress.x[-1])
    xs = np.linspace(float(progress.x[0]), T_end, 500)

    dprog = progress.derivative()

    all_cpu_funcs = sh_task._derived_cpu_funcs
    pc_req_rate = all_cpu_funcs[1]   # PageCache bw read
    disk_req_rate = all_cpu_funcs[3] # Disk bw read

    pc_consumed = []
    disk_consumed = []
    for xv in xs:
        dp = max(float(dprog(xv)), 0.0)
        prog_val = float(progress(xv))
        try:
            pc_r = _to_float(pc_req_rate(prog_val))
        except Exception:
            pc_r = 0.0
        try:
            dk_r = _to_float(disk_req_rate(prog_val))
        except Exception:
            dk_r = 0.0
        pc_consumed.append(dp * pc_r / 1e6)
        disk_consumed.append(dp * dk_r / 1e6)

    # Available (dashed)
    ax.axhline(y=mem_bw / 1e6, color=COLOR_MAP["memory_bw"], linestyle="--",
               linewidth=1.0, alpha=0.6)
    ax.axhline(y=disk_bw / 1e6, color=COLOR_MAP["disk_bw"], linestyle="--",
               linewidth=1.0, alpha=0.6)

    # Consumed (solid)
    ax.plot(xs, pc_consumed, color=COLOR_MAP["memory_bw"], linewidth=1.5,
            label="PageCache used")
    ax.plot(xs, disk_consumed, color=COLOR_MAP["disk_bw"], linewidth=1.5,
            label="Disk used")

    ax.set_xlabel("time (s)")
    max_bw = max(mem_bw, disk_bw) / 1e6
    ax.set_ylim(0, max_bw * 1.15)


def _plot_workflow_detail(
    *,
    out_path: Path,
    task_progresses: list[Any],
    task_bottlenecks: list[list[int]],
    task_sh_tasks: list[StorageHierarchyTask],
    task_labels: list[str],
    task_hit_rates: list[float],
    ordering_label: str,
    total_time_ca: float,
    total_time_meas: float | None,
    total_time_vanilla: float,
    disk_bw: float,
    mem_bw: float,
) -> None:
    """Paper-style 2x2 plot for a 4-task workflow.

    Top row:    progress (%) with bottleneck-coloured bands for each task
    Bottom row: resource consumption for each task

    Since we have 4 tasks, we use a 2x4 layout (top+bottom for each task).
    """
    _configure_matplotlib_paper_style()

    n_tasks = len(task_progresses)
    fig, axes = plt.subplots(
        2, n_tasks,
        figsize=(3.0 * n_tasks, 5.5),
        gridspec_kw={"height_ratios": [1.2, 1]},
    )

    for i in range(n_tasks):
        hit_pct = task_hit_rates[i] * 100
        title = f"{task_labels[i]}\nhit={hit_pct:.0f}%  t={float(task_progresses[i].x[-1]):.1f}s"

        _plot_progress_panel(
            axes[0, i], task_progresses[i], task_bottlenecks[i],
            task_sh_tasks[i], title,
        )
        if i == 0:
            axes[0, i].set_ylabel("progress (%)")
        axes[0, i].tick_params(axis="x", labelbottom=False)

        _plot_resource_panel(
            axes[1, i], task_progresses[i], task_sh_tasks[i], disk_bw, mem_bw,
        )
        if i == 0:
            axes[1, i].set_ylabel("bandwidth (MB/s)")

    # Suptitle
    meas_str = f"Measured: {total_time_meas:.1f}s" if total_time_meas else "Measured: N/A"
    fig.suptitle(
        f"{ordering_label}\n"
        f"CA: {total_time_ca:.1f}s  |  Vanilla: {total_time_vanilla:.1f}s  |  {meas_str}",
        fontsize=10, y=1.02,
    )

    # Shared legend
    legend_elems = [
        Patch(facecolor=COLOR_MAP["disk_bw"], alpha=0.25, label="Disk-bound"),
        Patch(facecolor=COLOR_MAP["memory_bw"], alpha=0.25, label="Memory-bound"),
        Line2D([0], [0], color=COLOR_MAP["disk_bw"], linewidth=1.5, label="Disk BW used"),
        Line2D([0], [0], color=COLOR_MAP["memory_bw"], linewidth=1.5, label="PageCache BW used"),
        Line2D([0], [0], color="gray", linewidth=1.0, linestyle="--", label="BW available"),
    ]
    axes[0, -1].legend(
        handles=legend_elems[:2], title="Bottleneck:", frameon=True,
        loc="lower right", fontsize=7,
    )
    axes[1, -1].legend(
        handles=legend_elems[2:], frameon=True, loc="upper right", fontsize=7,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_summary_comparison(
    *,
    out_path: Path,
    interleaved_meas_times: list[list[float]],
    grouped_meas_times: list[list[float]],
    interleaved_ca: float,
    grouped_ca: float,
    interleaved_vanilla: float,
    grouped_vanilla: float,
    mem_limit: int,
    file_size_a: int,
    file_size_b: int,
) -> None:
    """Summary bar chart: interleaved vs grouped, 3 models each."""
    _configure_matplotlib_paper_style()

    interleaved_totals = [sum(trial) for trial in interleaved_meas_times]
    grouped_totals = [sum(trial) for trial in grouped_meas_times]

    meas_interleaved_mean = float(np.mean(interleaved_totals))
    meas_grouped_mean = float(np.mean(grouped_totals))
    meas_interleaved_std = float(np.std(interleaved_totals))
    meas_grouped_std = float(np.std(grouped_totals))

    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    labels = ["Interleaved\n(A-B-A-B)", "Grouped\n(A-A-B-B)"]
    x = np.arange(len(labels))
    width = 0.22

    # Measured
    meas_vals = [meas_interleaved_mean, meas_grouped_mean]
    meas_errs = [meas_interleaved_std, meas_grouped_std]
    bars1 = ax.bar(
        x - width, meas_vals, width, yerr=meas_errs, capsize=4,
        color="black", alpha=0.7, label="Measured",
    )

    # BottleMod-CA
    ca_vals = [interleaved_ca, grouped_ca]
    bars2 = ax.bar(
        x, ca_vals, width, color="#F58518", alpha=0.8, label="BottleMod-CA",
    )

    # Vanilla
    vanilla_vals = [interleaved_vanilla, grouped_vanilla]
    bars3 = ax.bar(
        x + width, vanilla_vals, width, color="gray", alpha=0.6,
        label="Vanilla BottleMod",
    )

    # Value annotations
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}s",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    # Speedup annotation
    if meas_interleaved_mean > 0:
        speedup = meas_interleaved_mean / meas_grouped_mean
        ax.annotate(
            f"Grouped is {speedup:.2f}x faster",
            xy=(0.5, max(meas_vals) * 0.5),
            fontsize=10, ha="center", style="italic", color="#E45756",
        )

    ax.set_ylabel("Total workflow runtime (s)")
    ax.set_title(
        f"Task Ordering: Interleaved vs Grouped\n"
        f"Files: {_format_bytes_human(file_size_a)} each, "
        f"Memory: {_format_bytes_human(mem_limit)}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=True, loc="upper right")

    ax.set_ylim(0, max(meas_vals + ca_vals + vanilla_vals) * 1.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_per_task_breakdown(
    *,
    out_path: Path,
    interleaved_meas_per_task: list[list[float]],
    grouped_meas_per_task: list[list[float]],
    interleaved_ca_per_task: list[float],
    grouped_ca_per_task: list[float],
    interleaved_vanilla_per_task: float,
    grouped_vanilla_per_task: float,
    interleaved_hit_rates: list[float],
    grouped_hit_rates: list[float],
    mem_limit: int,
) -> None:
    """Per-task breakdown: stacked bars showing individual task times."""
    _configure_matplotlib_paper_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.5), sharey=True)

    task_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    interleaved_labels = ["Op1(A)", "Op1(B)", "Op2(A)", "Op2(B)"]
    grouped_labels = ["Op1(A)", "Op2(A)", "Op1(B)", "Op2(B)"]

    def _plot_stacked(ax, meas_per_task, ca_per_task, vanilla_per_task,
                      task_labels_list, hit_rates, title):
        """Plot stacked bars for measured, CA, vanilla."""
        categories = ["Measured\n(mean)", "BottleMod-CA", "Vanilla\nBottleMod"]
        x = np.arange(len(categories))
        width = 0.5

        # Compute measured means per task
        meas_means = [float(np.mean(t)) for t in meas_per_task]

        # Build stacked bars
        bottoms_meas = [0.0]
        bottoms_ca = [0.0]
        bottoms_vanilla = [0.0]

        for i in range(len(task_labels_list)):
            vals = [
                meas_means[i],
                ca_per_task[i],
                vanilla_per_task,
            ]
            bottoms = [
                sum(meas_means[:i]),
                sum(ca_per_task[:i]),
                vanilla_per_task * i,
            ]

            ax.bar(
                x, vals, width, bottom=bottoms,
                color=task_colors[i], alpha=0.75,
                label=f"{task_labels_list[i]} (hit={hit_rates[i]*100:.0f}%)",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_title(title)
        ax.legend(frameon=True, loc="upper right", fontsize=7)

    _plot_stacked(
        ax1,
        interleaved_meas_per_task,
        interleaved_ca_per_task,
        interleaved_vanilla_per_task,
        interleaved_labels,
        interleaved_hit_rates,
        "Interleaved (A-B-A-B)",
    )
    ax1.set_ylabel("Runtime (s)")

    _plot_stacked(
        ax2,
        grouped_meas_per_task,
        grouped_ca_per_task,
        grouped_vanilla_per_task,
        grouped_labels,
        grouped_hit_rates,
        "Grouped (A-A-B-B)",
    )

    fig.suptitle(
        f"Per-Task Breakdown  (Memory: {_format_bytes_human(mem_limit)})",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment 1: Cache-aware task ordering with two same-size files"
    )
    ap.add_argument("--video-a", required=True, help="Path to video file A")
    ap.add_argument("--video-b", required=True, help="Path to video file B (same size as A)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--mem-limit", default="5G", help="Fixed cgroup memory limit (default: 5G)")
    ap.add_argument("--trials", type=int, default=5, help="Number of measurement trials")
    ap.add_argument("--drop-caches", action="store_true", help="Drop caches before each trial")
    args = ap.parse_args()

    video_a = Path(args.video_a).resolve()
    video_b = Path(args.video_b).resolve()
    if not video_a.exists():
        print(f"Error: video A not found: {video_a}", file=sys.stderr)
        sys.exit(1)
    if not video_b.exists():
        print(f"Error: video B not found: {video_b}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mem_limit = _parse_size_bytes(args.mem_limit)
    file_size_a = video_a.stat().st_size
    file_size_b = video_b.stat().st_size

    print(f"Video A: {video_a} ({_format_bytes_human(file_size_a)})")
    print(f"Video B: {video_b} ({_format_bytes_human(file_size_b)})")
    print(f"Memory limit: {_format_bytes_human(mem_limit)}")
    print(f"Trials: {args.trials}")
    print()

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    disk_bw, mem_bw, _ = calibrate(
        video_a, out_dir, drop_caches=args.drop_caches, mem_limit=mem_limit,
    )

    # ------------------------------------------------------------------
    # Compute hit rates
    # ------------------------------------------------------------------
    interleaved_hits = compute_hit_rates_interleaved(
        float(file_size_a), float(file_size_b), mem_limit,
    )
    grouped_hits = compute_hit_rates_grouped(
        float(file_size_a), float(file_size_b), mem_limit,
    )

    print(f"\nInterleaved hit rates: {[f'{h:.3f}' for h in interleaved_hits]}")
    print(f"Grouped hit rates:     {[f'{h:.3f}' for h in grouped_hits]}")

    # ------------------------------------------------------------------
    # Model predictions
    # ------------------------------------------------------------------
    # File sizes for each task position
    interleaved_files = [file_size_a, file_size_b, file_size_a, file_size_b]
    grouped_files = [file_size_a, file_size_a, file_size_b, file_size_b]

    interleaved_labels = ["Op1(A)", "Op1(B)", "Op2(A)", "Op2(B)"]
    grouped_labels = ["Op1(A)", "Op2(A)", "Op1(B)", "Op2(B)"]

    # Vanilla: same prediction for every task (no cache model)
    vanilla_time_a = predict_vanilla(float(file_size_a), disk_bw)
    vanilla_time_b = predict_vanilla(float(file_size_b), disk_bw)

    vanilla_interleaved_total = 2 * vanilla_time_a + 2 * vanilla_time_b
    vanilla_grouped_total = 2 * vanilla_time_a + 2 * vanilla_time_b

    print(f"\nVanilla per-task: A={vanilla_time_a:.2f}s, B={vanilla_time_b:.2f}s")
    print(f"Vanilla total (both orderings): {vanilla_interleaved_total:.2f}s")

    # CA predictions
    print("\n--- BottleMod-CA Interleaved ---")
    ca_interleaved_times: list[float] = []
    ca_interleaved_progresses: list[Any] = []
    ca_interleaved_bottlenecks: list[list[int]] = []
    ca_interleaved_sh_tasks: list[StorageHierarchyTask] = []

    for i in range(4):
        t, prog, bn, sh = predict_cache_aware(
            float(interleaved_files[i]), disk_bw, mem_bw, interleaved_hits[i],
        )
        ca_interleaved_times.append(t)
        ca_interleaved_progresses.append(prog)
        ca_interleaved_bottlenecks.append(bn)
        ca_interleaved_sh_tasks.append(sh)
        print(f"  {interleaved_labels[i]}: hit={interleaved_hits[i]:.3f}, t={t:.2f}s")

    ca_interleaved_total = sum(ca_interleaved_times)
    print(f"  Total: {ca_interleaved_total:.2f}s")

    print("\n--- BottleMod-CA Grouped ---")
    ca_grouped_times: list[float] = []
    ca_grouped_progresses: list[Any] = []
    ca_grouped_bottlenecks: list[list[int]] = []
    ca_grouped_sh_tasks: list[StorageHierarchyTask] = []

    for i in range(4):
        t, prog, bn, sh = predict_cache_aware(
            float(grouped_files[i]), disk_bw, mem_bw, grouped_hits[i],
        )
        ca_grouped_times.append(t)
        ca_grouped_progresses.append(prog)
        ca_grouped_bottlenecks.append(bn)
        ca_grouped_sh_tasks.append(sh)
        print(f"  {grouped_labels[i]}: hit={grouped_hits[i]:.3f}, t={t:.2f}s")

    ca_grouped_total = sum(ca_grouped_times)
    print(f"  Total: {ca_grouped_total:.2f}s")

    # ------------------------------------------------------------------
    # Measurements
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Running measurements")
    print(f"{'='*60}")

    interleaved_all_trials: list[list[float]] = []
    grouped_all_trials: list[list[float]] = []

    for trial in range(1, args.trials + 1):
        print(f"\n--- Trial {trial}/{args.trials}: Interleaved ---")
        times, total = _run_four_task_workflow(
            video_a, video_b, out_dir,
            ordering="interleaved", mem_limit=mem_limit,
            drop_caches=args.drop_caches,
        )
        interleaved_all_trials.append(times)
        for i, t in enumerate(times):
            print(f"  {interleaved_labels[i]}: {t:.2f}s")
        print(f"  Total: {total:.2f}s")

    for trial in range(1, args.trials + 1):
        print(f"\n--- Trial {trial}/{args.trials}: Grouped ---")
        times, total = _run_four_task_workflow(
            video_a, video_b, out_dir,
            ordering="grouped", mem_limit=mem_limit,
            drop_caches=args.drop_caches,
        )
        grouped_all_trials.append(times)
        for i, t in enumerate(times):
            print(f"  {grouped_labels[i]}: {t:.2f}s")
        print(f"  Total: {total:.2f}s")

    # Compute per-task stats
    interleaved_per_task = [
        [trial[i] for trial in interleaved_all_trials] for i in range(4)
    ]
    grouped_per_task = [
        [trial[i] for trial in grouped_all_trials] for i in range(4)
    ]

    interleaved_totals = [sum(t) for t in interleaved_all_trials]
    grouped_totals = [sum(t) for t in grouped_all_trials]

    meas_interleaved_mean = float(np.mean(interleaved_totals))
    meas_grouped_mean = float(np.mean(grouped_totals))

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Interleaved  measured: {meas_interleaved_mean:.2f}s  "
          f"(std={float(np.std(interleaved_totals)):.2f}s)")
    print(f"Grouped      measured: {meas_grouped_mean:.2f}s  "
          f"(std={float(np.std(grouped_totals)):.2f}s)")
    print(f"Interleaved  CA pred:  {ca_interleaved_total:.2f}s")
    print(f"Grouped      CA pred:  {ca_grouped_total:.2f}s")
    print(f"Vanilla pred (both):   {vanilla_interleaved_total:.2f}s")
    if meas_grouped_mean > 0:
        print(f"Speedup (grouped):     {meas_interleaved_mean / meas_grouped_mean:.2f}x")

    # ------------------------------------------------------------------
    # Write results JSON
    # ------------------------------------------------------------------
    results: dict[str, Any] = {
        "schema_version": 4,
        "meta": {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "video_a": str(video_a),
            "video_b": str(video_b),
            "file_size_a_bytes": file_size_a,
            "file_size_b_bytes": file_size_b,
            "mem_limit_bytes": mem_limit,
            "mem_limit_label": _format_bytes(mem_limit),
            "trials": args.trials,
            "drop_caches": args.drop_caches,
            "disk_bw_bytes_s": disk_bw,
            "mem_bw_bytes_s": mem_bw,
        },
        "interleaved": {
            "hit_rates": interleaved_hits,
            "ca_per_task_s": ca_interleaved_times,
            "ca_total_s": ca_interleaved_total,
            "vanilla_per_task_s": vanilla_time_a,  # same for all
            "vanilla_total_s": vanilla_interleaved_total,
            "measured_trials": interleaved_all_trials,
            "measured_per_task_mean_s": [float(np.mean(t)) for t in interleaved_per_task],
            "measured_total_mean_s": meas_interleaved_mean,
            "measured_total_std_s": float(np.std(interleaved_totals)),
        },
        "grouped": {
            "hit_rates": grouped_hits,
            "ca_per_task_s": ca_grouped_times,
            "ca_total_s": ca_grouped_total,
            "vanilla_per_task_s": vanilla_time_a,
            "vanilla_total_s": vanilla_grouped_total,
            "measured_trials": grouped_all_trials,
            "measured_per_task_mean_s": [float(np.mean(t)) for t in grouped_per_task],
            "measured_total_mean_s": meas_grouped_mean,
            "measured_total_std_s": float(np.std(grouped_totals)),
        },
    }

    out_json = out_dir / "exp1_reordering_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to: {out_json}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    # Fig 1: Interleaved workflow detail (2x4 progress + resource)
    fig1_path = out_dir / "fig_interleaved_detail.png"
    _plot_workflow_detail(
        out_path=fig1_path,
        task_progresses=ca_interleaved_progresses,
        task_bottlenecks=ca_interleaved_bottlenecks,
        task_sh_tasks=ca_interleaved_sh_tasks,
        task_labels=interleaved_labels,
        task_hit_rates=interleaved_hits,
        ordering_label="Interleaved: Op1(A) -> Op1(B) -> Op2(A) -> Op2(B)",
        total_time_ca=ca_interleaved_total,
        total_time_meas=meas_interleaved_mean,
        total_time_vanilla=vanilla_interleaved_total,
        disk_bw=disk_bw,
        mem_bw=mem_bw,
    )
    print(f"Figure (interleaved detail): {fig1_path}")

    # Fig 2: Grouped workflow detail (2x4 progress + resource)
    fig2_path = out_dir / "fig_grouped_detail.png"
    _plot_workflow_detail(
        out_path=fig2_path,
        task_progresses=ca_grouped_progresses,
        task_bottlenecks=ca_grouped_bottlenecks,
        task_sh_tasks=ca_grouped_sh_tasks,
        task_labels=grouped_labels,
        task_hit_rates=grouped_hits,
        ordering_label="Grouped: Op1(A) -> Op2(A) -> Op1(B) -> Op2(B)",
        total_time_ca=ca_grouped_total,
        total_time_meas=meas_grouped_mean,
        total_time_vanilla=vanilla_grouped_total,
        disk_bw=disk_bw,
        mem_bw=mem_bw,
    )
    print(f"Figure (grouped detail): {fig2_path}")

    # Fig 3: Summary comparison bar chart
    fig3_path = out_dir / "fig_summary_comparison.png"
    _plot_summary_comparison(
        out_path=fig3_path,
        interleaved_meas_times=interleaved_all_trials,
        grouped_meas_times=grouped_all_trials,
        interleaved_ca=ca_interleaved_total,
        grouped_ca=ca_grouped_total,
        interleaved_vanilla=vanilla_interleaved_total,
        grouped_vanilla=vanilla_grouped_total,
        mem_limit=mem_limit,
        file_size_a=file_size_a,
        file_size_b=file_size_b,
    )
    print(f"Figure (summary comparison): {fig3_path}")

    # Fig 4: Per-task breakdown
    fig4_path = out_dir / "fig_per_task_breakdown.png"
    _plot_per_task_breakdown(
        out_path=fig4_path,
        interleaved_meas_per_task=interleaved_per_task,
        grouped_meas_per_task=grouped_per_task,
        interleaved_ca_per_task=ca_interleaved_times,
        grouped_ca_per_task=ca_grouped_times,
        interleaved_vanilla_per_task=vanilla_time_a,
        grouped_vanilla_per_task=vanilla_time_a,
        interleaved_hit_rates=interleaved_hits,
        grouped_hit_rates=grouped_hits,
        mem_limit=mem_limit,
    )
    print(f"Figure (per-task breakdown): {fig4_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
