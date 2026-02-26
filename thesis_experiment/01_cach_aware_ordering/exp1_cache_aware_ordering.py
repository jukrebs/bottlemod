#!/usr/bin/env python3
"""
Experiment 1 — Cache-Aware Modeling of FFmpeg Video Remux

Compares vanilla BottleMod (no cache awareness) against BottleMod-CA
(cache-aware) for an ffmpeg video remux workload under varying cgroup
memory limits.  Optionally demonstrates task reordering with two videos.

Usage:
    .venv/bin/python thesis_experiment/01_cach_aware_ordering/exp1_cache_aware_ordering.py \
        --video /mnt/sata/input.mp4 \
        --video2 /mnt/sata/input_small.mp4 \
        --out-dir /var/tmp/exp1_ffmpeg \
        --mem-sweep 256M,512M,1G,2G,4G,8G,16G \
        --trials 5 --drop-caches
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
from typing import Any, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from bottlemod.func import Func

# ---------------------------------------------------------------------------
# BottleMod-CA imports (package install)
# ---------------------------------------------------------------------------
from bottlemod.ppoly import PPoly
from bottlemod.storage_hierarchy import (
    LogicalAccessProfile,
    StorageHierarchyTask,
    StorageTier,
    TierMapping,
    get_bottleneck_label,
)
from bottlemod.task import TaskExecution

# ---------------------------------------------------------------------------
# Vanilla BottleMod imports (via sys.path for relative-import package)
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
    if n >= 1024**3 and n % (1024**3) == 0:
        return f"{n // (1024**3)}G"
    if n >= 1024**2 and n % (1024**2) == 0:
        return f"{n // (1024**2)}M"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return f"{n}B"


def _format_bytes_human(n: int) -> str:
    """Human-readable size with one decimal (e.g. '4.3 GB')."""
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
    """Run ffmpeg remux and return wall-clock time in seconds.

    If *mem_limit* is set, run inside a systemd-run cgroup with that MemoryMax.
    """
    if drop_caches:
        _sudo_drop_caches()

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(video_in),
        "-c",
        "copy",
        str(video_out),
    ]

    if mem_limit is not None:
        cmd = [
            "sudo",
            "systemd-run",
            "--wait",
            "--collect",
            "--quiet",
            f"--property=MemoryMax={mem_limit}",
            "--property=MemorySwapMax=0",
            "--",
        ] + ffmpeg_cmd
    else:
        cmd = ffmpeg_cmd

    t0 = time.monotonic()
    _run(cmd, check=True, capture=True)
    t1 = time.monotonic()

    # Clean up output file to avoid filling disk
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
    cold_mem_limit: int,
    warm_mem_limit: int,
) -> Tuple[float, float, int]:
    """Measure cold (disk) and warm (cache) bandwidth under cgroup constraints.

    Cold run uses *cold_mem_limit* (smallest cgroup) so that the measured
    disk bandwidth reflects the I/O amplification caused by memory pressure.
    Warm run uses *warm_mem_limit* (largest cgroup) so the file fits in cache.

    Returns ``(disk_bw_bytes_s, mem_bw_bytes_s, file_size_bytes)``.
    """
    file_size = video.stat().st_size
    tmp_out = out_dir / "calib_output.mp4"

    print(f"Calibrating: file_size = {file_size / 1e9:.2f} GB")

    cold_time = _run_ffmpeg_remux(
        video, tmp_out, mem_limit=cold_mem_limit, drop_caches=drop_caches
    )
    disk_bw = file_size / max(cold_time, 1e-9)
    print(
        f"  Cold run (cgroup {_format_bytes(cold_mem_limit)}): "
        f"{cold_time:.2f}s  ->  disk_bw = {disk_bw / 1e6:.1f} MB/s"
    )

    # Warm: pre-load file into page cache, then measure with generous cgroup
    _run_ffmpeg_remux(video, tmp_out)
    warm_time = _run_ffmpeg_remux(video, tmp_out, mem_limit=warm_mem_limit)
    mem_bw = file_size / max(warm_time, 1e-9)
    print(
        f"  Warm run (cgroup {_format_bytes(warm_mem_limit)}): "
        f"{warm_time:.2f}s  ->  mem_bw  = {mem_bw / 1e6:.1f} MB/s"
    )

    return disk_bw, mem_bw, file_size


# ===========================================================================
# Modeling
# ===========================================================================


def predict_vanilla(file_size: float, disk_bw: float) -> float:
    """Predict runtime using vanilla BottleMod (no cache awareness).

    Vanilla BottleMod has no concept of storage tiers, so its prediction
    is a constant ``file_size / disk_bw`` regardless of cache state.
    """
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
    file_size: float, disk_bw: float, mem_bw: float, hit_rate: float
) -> StorageHierarchyTask:
    """Create a StorageHierarchyTask for the given parameters."""
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
                name="PageCache", tier_index=0, I_bw_read=PPoly([0, T_max], [[mem_bw]])
            ),
            StorageTier(
                name="Disk", tier_index=1, I_bw_read=PPoly([0, T_max], [[disk_bw]])
            ),
        ],
        cpu_funcs=[PPoly([0, max_progress], [[1e-9]])],
        data_funcs=[Func([0, max_progress], [[1, 0]])],
    )


def predict_cache_aware(
    file_size: float, disk_bw: float, mem_bw: float, hit_rate: float
) -> Tuple[float, Any, Any, StorageHierarchyTask]:
    """Predict runtime using BottleMod-CA.

    Returns ``(predicted_time, progress_func, bottleneck_list, sh_task)``.
    """
    max_progress = 1.0
    T_max = file_size / disk_bw * 2.0

    sh_task = _make_sh_task(file_size, disk_bw, mem_bw, hit_rate)

    bm_task = sh_task.to_task()
    storage_inputs = sh_task.get_storage_input_funcs()

    in_cpu = [PPoly([0, T_max], [[1e12]])]
    all_in_cpu = in_cpu + storage_inputs
    in_data_func = Func([0, T_max], [[max_progress]])
    in_data = [in_data_func]

    execution = TaskExecution(bm_task, all_in_cpu, in_data)
    progress, bottlenecks = execution.get_result()
    return float(progress.x[-1]), progress, bottlenecks, sh_task


# ===========================================================================
# Plotting
# ===========================================================================


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


COLOR_MAP = {
    "disk_bw": "#F58518",  # orange
    "memory_bw": "#4C78A8",  # blue
    "cpu": "#54A24B",  # green
    "data": "#999999",  # gray
}


def _bottleneck_category(bn_index: int, sh_task: StorageHierarchyTask) -> str:
    """Map a bottleneck index to a colour-map category string."""
    label = get_bottleneck_label(bn_index, sh_task).lower()
    if "pagecache" in label or "mem" in label or "cache" in label:
        return "memory_bw"
    elif "disk" in label:
        return "disk_bw"
    elif "cpu" in label:
        return "cpu"
    return "data"


def _plot_fig6(
    *,
    out_path: Path,
    mem_labels: list[str],
    vanilla_pred: list[float],
    ca_pred: list[float],
    mean_s: list[float],
    min_s: list[float],
    max_s: list[float],
    file_size_bytes: int,
) -> None:
    """Paper Figure-6 style: prediction vs measurement."""
    _configure_matplotlib_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.5))

    x = list(range(len(mem_labels)))

    # Vanilla prediction (flat dashed)
    ax.plot(
        x,
        vanilla_pred,
        color="gray",
        linewidth=1.5,
        linestyle="--",
        label="Vanilla BottleMod",
    )

    # BottleMod-CA prediction
    ax.plot(
        x,
        ca_pred,
        color="#F58518",
        linewidth=2.0,
        label="BottleMod-CA prediction",
    )

    # Measured
    yerr = np.array(
        [
            np.array(mean_s) - np.array(min_s),
            np.array(max_s) - np.array(mean_s),
        ]
    )
    ax.errorbar(
        x,
        mean_s,
        yerr=yerr,
        fmt="o",
        color="black",
        markersize=4,
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
        label="Measured (mean +/- min/max)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(mem_labels, rotation=45, ha="right")
    ax.set_xlabel("Available system memory (cgroup limit)")
    ax.set_ylabel("Runtime (s)")
    ax.set_title(
        f"FFmpeg Remux: Prediction vs Measurement  "
        f"(file size: {_format_bytes_human(file_size_bytes)})"
    )

    # Annotate file size as vertical reference line
    file_gb = file_size_bytes / (1024**3)
    for i, lbl in enumerate(mem_labels):
        mem_bytes = _parse_size_bytes(lbl)
        if mem_bytes / (1024**3) >= file_gb:
            ax.axvline(
                x=i,
                color="#E45756",
                linewidth=1.0,
                linestyle=":",
                alpha=0.7,
            )
            ax.annotate(
                f"file size\n({file_gb:.1f} GB)",
                xy=(i, ax.get_ylim()[1]),
                xytext=(i + 0.15, ax.get_ylim()[1] * 0.95),
                fontsize=8,
                color="#E45756",
                ha="left",
                va="top",
            )
            break

    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_fig7(
    *,
    out_path: Path,
    cold_progress: Any,
    cold_bottlenecks: list[int],
    cold_task: StorageHierarchyTask,
    warm_progress: Any,
    warm_bottlenecks: list[int],
    warm_task: StorageHierarchyTask,
    disk_bw: float,
    mem_bw: float,
) -> None:
    """Paper Figure-7 style: 2x2 bottleneck timeline + resource consumption.

    Top row:    progress (%) over time with bottleneck-coloured bands
    Bottom row: resource consumption rates (bandwidth used vs available)
    Left:       cold cache (low memory)
    Right:      warm cache (high memory)
    """
    _configure_matplotlib_paper_style()

    fig, ((ax_prog_cold, ax_prog_warm), (ax_res_cold, ax_res_warm)) = plt.subplots(
        2,
        2,
        figsize=(9.0, 5.5),
        gridspec_kw={"height_ratios": [1.2, 1]},
    )

    def _plot_progress_panel(ax, progress, bottlenecks, sh_task, title):
        T_end = float(progress.x[-1])
        for i, seg_x in enumerate(progress.x[:-1]):
            seg_end = progress.x[i + 1]
            if i < len(bottlenecks):
                cat = _bottleneck_category(bottlenecks[i], sh_task)
                color = COLOR_MAP.get(cat, "#999999")
            else:
                color = "#999999"
            ax.axvspan(
                float(seg_x), float(seg_end), color=color, alpha=0.25, linewidth=0
            )

        xs = np.linspace(float(progress.x[0]), T_end, 500)
        max_val = float(progress(T_end))
        ys = [
            float(progress(xv)) / max_val * 100.0 if max_val > 0 else 0.0 for xv in xs
        ]
        ax.plot(xs, ys, color="#4C78A8", linewidth=1.8)
        ax.set_ylim(0, 105)
        ax.set_title(title)

    def _plot_resource_panel(
        ax, progress, bottlenecks, sh_task, disk_bw_val, mem_bw_val
    ):
        """Plot bandwidth usage per tier over time (like paper's bottom row).

        For each segment, the consumed bandwidth of a tier equals:
            consumed_bw = progress_derivative(t) * tier_requirement_rate(progress(t))

        We show:
          - dashed line: available bandwidth (constant, I_{R,l})
          - solid line:  consumed bandwidth
        """
        T_end = float(progress.x[-1])
        xs = np.linspace(float(progress.x[0]), T_end, 500)

        dprog = progress.derivative()

        # Tier resource functions come from sh_task
        # Resource order: [CPU_0, PageCache_bw_read, PageCache_bw_write, Disk_bw_read, Disk_bw_write]
        # We care about bw_read for PageCache (index 1) and Disk (index 3)
        all_cpu_funcs = sh_task._derived_cpu_funcs  # [cpu, pc_r, pc_w, disk_r, disk_w]
        pc_req_rate = all_cpu_funcs[1]  # R'_{PageCache,bw,r}(p) = H_cache * A'_read
        disk_req_rate = all_cpu_funcs[3]  # R'_{Disk,bw,r}(p) = H_disk * A'_read

        # Consumed bandwidth at time t = dprog/dt * R'(progress(t))
        pc_consumed = []
        disk_consumed = []
        for xv in xs:
            dp = max(float(dprog(xv)), 0.0)
            prog_val = float(progress(xv))
            try:
                pc_r = float(pc_req_rate(prog_val))
            except Exception:
                pc_r = 0.0
            try:
                dk_r = float(disk_req_rate(prog_val))
            except Exception:
                dk_r = 0.0
            pc_consumed.append(dp * pc_r / 1e6)  # MB/s
            disk_consumed.append(dp * dk_r / 1e6)  # MB/s

        # Available (constant lines)
        ax.axhline(
            y=mem_bw_val / 1e6,
            color=COLOR_MAP["memory_bw"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )
        ax.axhline(
            y=disk_bw_val / 1e6,
            color=COLOR_MAP["disk_bw"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )

        # Consumed
        ax.plot(
            xs,
            pc_consumed,
            color=COLOR_MAP["memory_bw"],
            linewidth=1.5,
            label="PageCache used",
        )
        ax.plot(
            xs,
            disk_consumed,
            color=COLOR_MAP["disk_bw"],
            linewidth=1.5,
            label="Disk used",
        )

        ax.set_xlabel("time (s)")
        max_bw = max(mem_bw_val, disk_bw_val) / 1e6
        ax.set_ylim(0, max_bw * 1.15)

    # Top row: progress
    _plot_progress_panel(
        ax_prog_cold,
        cold_progress,
        cold_bottlenecks,
        cold_task,
        "Cold cache (low memory)",
    )
    ax_prog_cold.set_ylabel("progress (%)")
    ax_prog_cold.tick_params(axis="x", labelbottom=False)

    _plot_progress_panel(
        ax_prog_warm,
        warm_progress,
        warm_bottlenecks,
        warm_task,
        "Warm cache (high memory)",
    )
    ax_prog_warm.tick_params(axis="x", labelbottom=False)

    # Bottom row: resource consumption
    _plot_resource_panel(
        ax_res_cold, cold_progress, cold_bottlenecks, cold_task, disk_bw, mem_bw
    )
    ax_res_cold.set_ylabel("bandwidth (MB/s)")

    _plot_resource_panel(
        ax_res_warm, warm_progress, warm_bottlenecks, warm_task, disk_bw, mem_bw
    )

    # Shared legend
    legend_elems = [
        Patch(facecolor=COLOR_MAP["disk_bw"], alpha=0.25, label="Disk-bound"),
        Patch(facecolor=COLOR_MAP["memory_bw"], alpha=0.25, label="Memory-bound"),
        Line2D(
            [0], [0], color=COLOR_MAP["disk_bw"], linewidth=1.5, label="Disk BW used"
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_MAP["memory_bw"],
            linewidth=1.5,
            label="PageCache BW used",
        ),
        Line2D(
            [0], [0], color="gray", linewidth=1.0, linestyle="--", label="BW available"
        ),
    ]
    ax_prog_cold.legend(
        handles=legend_elems[:2],
        title="Bottleneck:",
        frameon=True,
        loc="lower right",
        fontsize=8,
    )
    ax_res_cold.legend(
        handles=legend_elems[2:], frameon=True, loc="upper right", fontsize=8
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ===========================================================================
# Two-video workflow: task reordering experiment
# ===========================================================================


def _predict_two_video_workflow_ca(
    file_size_a: float,
    file_size_b: float,
    disk_bw: float,
    mem_bw: float,
    mem_limit: int,
    order: str,
) -> Tuple[float, Any, list, Any, list, StorageHierarchyTask, StorageHierarchyTask]:
    """Model two sequential ffmpeg remux tasks under a shared memory limit.

    *order*: "AB" (A first then B) or "BA" (B first then A).

    After the first task completes, its pages may remain in cache, affecting
    the hit rate of the second task.  For simplicity we assume sequential
    (no-reuse) access: pages from task 1 partially evict task 2's working set.

    Returns (total_time, prog1, bn1, prog2, bn2, sh_task1, sh_task2).
    """
    if order == "AB":
        size1, size2 = file_size_a, file_size_b
    else:
        size1, size2 = file_size_b, file_size_a

    # Task 1: cold start (drop caches assumed), hit_rate = mem_limit / size1
    hit1 = min(1.0, mem_limit / size1)
    t1, prog1, bn1, sh1 = predict_cache_aware(size1, disk_bw, mem_bw, hit1)

    # Task 2: after task 1, the page cache holds pages from task 1's file.
    # If mem_limit > size1, there's (mem_limit - size1) bytes of free cache
    # for task 2.  If mem_limit <= size1, cache is full of task 1 pages which
    # are useless for task 2 -> cold start.
    remaining_cache = max(0, mem_limit - size1)
    hit2 = min(1.0, remaining_cache / size2)
    t2, prog2, bn2, sh2 = predict_cache_aware(size2, disk_bw, mem_bw, hit2)

    return (t1 + t2, prog1, bn1, prog2, bn2, sh1, sh2)


def _run_two_video_sequential(
    video1: Path,
    video2: Path,
    out_dir: Path,
    *,
    mem_limit: int,
    drop_caches: bool,
) -> float:
    """Run two ffmpeg remux tasks sequentially under one cgroup memory limit.

    Returns total wall-clock time for both tasks.
    """
    tmp_out = out_dir / "tmp_workflow_output.mp4"

    if drop_caches:
        _sudo_drop_caches()

    # Task 1
    t1 = _run_ffmpeg_remux(video1, tmp_out, mem_limit=mem_limit)
    # Task 2 (don't drop caches between -- cache state from task 1 persists)
    t2 = _run_ffmpeg_remux(video2, tmp_out, mem_limit=mem_limit)

    return t1 + t2


def _plot_fig8_reordering(
    *,
    out_path: Path,
    mem_labels: list[str],
    pred_ab: list[float],
    pred_ba: list[float],
    meas_ab: list[float],
    meas_ba: list[float],
    file_size_a: int,
    file_size_b: int,
) -> None:
    """Figure 8: two-video workflow -- task ordering comparison.

    Shows that processing the smaller video first can be faster because its
    pages leave more cache room for the larger video.
    """
    _configure_matplotlib_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    x = list(range(len(mem_labels)))
    width = 0.18

    # Predictions
    ax.plot(
        x,
        pred_ab,
        color="#F58518",
        linewidth=2.0,
        linestyle="-",
        marker="^",
        markersize=5,
        label="CA pred: A->B",
    )
    ax.plot(
        x,
        pred_ba,
        color="#4C78A8",
        linewidth=2.0,
        linestyle="-",
        marker="v",
        markersize=5,
        label="CA pred: B->A",
    )

    # Measurements
    ax.bar(
        [xi - width for xi in x],
        meas_ab,
        width=width * 2,
        color="#F58518",
        alpha=0.3,
        label="Measured A->B",
    )
    ax.bar(
        [xi + width for xi in x],
        meas_ba,
        width=width * 2,
        color="#4C78A8",
        alpha=0.3,
        label="Measured B->A",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(mem_labels, rotation=45, ha="right")
    ax.set_xlabel("Available system memory (cgroup limit)")
    ax.set_ylabel("Total workflow runtime (s)")
    ax.set_title(
        f"Two-Video Workflow: Task Ordering\n"
        f"A = {_format_bytes_human(file_size_a)}, "
        f"B = {_format_bytes_human(file_size_b)}"
    )
    ax.legend(frameon=True, loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_fig9_bottleneck_reorder(
    *,
    out_path: Path,
    prog1_best: Any,
    bn1_best: list,
    sh1_best: StorageHierarchyTask,
    prog2_best: Any,
    bn2_best: list,
    sh2_best: StorageHierarchyTask,
    prog1_worst: Any,
    bn1_worst: list,
    sh1_worst: StorageHierarchyTask,
    prog2_worst: Any,
    bn2_worst: list,
    sh2_worst: StorageHierarchyTask,
    best_label: str,
    worst_label: str,
    disk_bw: float,
    mem_bw: float,
) -> None:
    """Figure 9: 2x2 progress + bottleneck timeline for best vs worst ordering.

    Left column:  best ordering (e.g. B->A)
    Right column: worst ordering (e.g. A->B)
    Top row:      task 1 progress
    Bottom row:   task 2 progress
    """
    _configure_matplotlib_paper_style()

    fig, ((ax_t1_best, ax_t1_worst), (ax_t2_best, ax_t2_worst)) = plt.subplots(
        2,
        2,
        figsize=(9.0, 5.5),
    )

    def _plot_panel(ax, progress, bottlenecks, sh_task, title):
        T_end = float(progress.x[-1])
        for i, seg_x in enumerate(progress.x[:-1]):
            seg_end = progress.x[i + 1]
            if i < len(bottlenecks):
                cat = _bottleneck_category(bottlenecks[i], sh_task)
                color = COLOR_MAP.get(cat, "#999999")
            else:
                color = "#999999"
            ax.axvspan(
                float(seg_x), float(seg_end), color=color, alpha=0.25, linewidth=0
            )

        xs = np.linspace(float(progress.x[0]), T_end, 500)
        max_val = float(progress(T_end))
        ys = [
            float(progress(xv)) / max_val * 100.0 if max_val > 0 else 0.0 for xv in xs
        ]
        ax.plot(xs, ys, color="#4C78A8", linewidth=1.8)
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=9)

    t1_best_time = float(prog1_best.x[-1])
    t2_best_time = float(prog2_best.x[-1])
    t1_worst_time = float(prog1_worst.x[-1])
    t2_worst_time = float(prog2_worst.x[-1])

    _plot_panel(
        ax_t1_best,
        prog1_best,
        bn1_best,
        sh1_best,
        f"{best_label}: Task 1 ({t1_best_time:.1f}s)",
    )
    ax_t1_best.set_ylabel("progress (%)")
    ax_t1_best.tick_params(axis="x", labelbottom=False)

    _plot_panel(
        ax_t1_worst,
        prog1_worst,
        bn1_worst,
        sh1_worst,
        f"{worst_label}: Task 1 ({t1_worst_time:.1f}s)",
    )
    ax_t1_worst.tick_params(axis="x", labelbottom=False)

    _plot_panel(
        ax_t2_best,
        prog2_best,
        bn2_best,
        sh2_best,
        f"{best_label}: Task 2 ({t2_best_time:.1f}s)",
    )
    ax_t2_best.set_ylabel("progress (%)")
    ax_t2_best.set_xlabel("time (s)")

    _plot_panel(
        ax_t2_worst,
        prog2_worst,
        bn2_worst,
        sh2_worst,
        f"{worst_label}: Task 2 ({t2_worst_time:.1f}s)",
    )
    ax_t2_worst.set_xlabel("time (s)")

    # Suptitle with totals
    fig.suptitle(
        f"Best order: {best_label} = {t1_best_time + t2_best_time:.1f}s total  |  "
        f"Worst order: {worst_label} = {t1_worst_time + t2_worst_time:.1f}s total",
        fontsize=10,
        y=1.01,
    )

    legend_elems = [
        Patch(facecolor=COLOR_MAP["disk_bw"], alpha=0.25, label="Disk-bound"),
        Patch(facecolor=COLOR_MAP["memory_bw"], alpha=0.25, label="Memory-bound"),
    ]
    ax_t1_best.legend(
        handles=legend_elems,
        title="Bottleneck:",
        frameon=True,
        loc="lower right",
        fontsize=7,
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
        description="Experiment 1: BottleMod-CA vs vanilla for ffmpeg remux"
    )
    ap.add_argument("--video", required=True, help="Path to input video file (video A)")
    ap.add_argument(
        "--video2",
        default=None,
        help="Path to second video file (video B) for reordering experiment",
    )
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument(
        "--mem-sweep",
        default="256M,512M,1G,2G,4G,8G,16G",
        help="Comma-separated cgroup memory limits",
    )
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--drop-caches", action="store_true")
    args = ap.parse_args()

    video = Path(args.video).resolve()
    if not video.exists():
        print(f"Error: video file not found: {video}", file=sys.stderr)
        sys.exit(1)

    video2 = None
    if args.video2:
        video2 = Path(args.video2).resolve()
        if not video2.exists():
            print(f"Error: video2 file not found: {video2}", file=sys.stderr)
            sys.exit(1)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mem_sweep = [_parse_size_bytes(x) for x in args.mem_sweep.split(",") if x.strip()]

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    disk_bw, mem_bw, file_size = calibrate(
        video,
        out_dir,
        drop_caches=args.drop_caches,
        cold_mem_limit=min(mem_sweep),
        warm_mem_limit=max(mem_sweep),
    )

    file_size2 = video2.stat().st_size if video2 else 0

    # ------------------------------------------------------------------
    # Measurement + prediction sweep (single video)
    # ------------------------------------------------------------------
    results: dict[str, Any] = {
        "schema_version": 3,
        "meta": {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "video": str(video),
            "file_size_bytes": file_size,
            "video2": str(video2) if video2 else None,
            "file_size2_bytes": file_size2 if video2 else None,
            "mem_sweep_bytes": mem_sweep,
            "trials": args.trials,
            "drop_caches": args.drop_caches,
            "disk_bw_bytes_s": disk_bw,
            "mem_bw_bytes_s": mem_bw,
        },
        "points": [],
        "reorder": [],
    }

    tmp_out = out_dir / "tmp_output.mp4"

    for mem_limit in mem_sweep:
        label = _format_bytes(mem_limit)
        hit_rate = min(1.0, mem_limit / file_size)

        # Predictions
        vanilla_time = predict_vanilla(float(file_size), disk_bw)
        ca_time, _, _, _ = predict_cache_aware(
            float(file_size), disk_bw, mem_bw, hit_rate
        )

        print(f"\nMemory limit: {label}  hit_rate={hit_rate:.3f}")
        print(f"  Vanilla prediction: {vanilla_time:.2f}s")
        print(f"  CA prediction:      {ca_time:.2f}s")

        # Measurements
        trial_times: list[float] = []
        for trial in range(1, args.trials + 1):
            t = _run_ffmpeg_remux(
                video,
                tmp_out,
                mem_limit=mem_limit,
                drop_caches=args.drop_caches,
            )
            trial_times.append(t)
            print(f"  Trial {trial}: {t:.2f}s")

        results["points"].append(
            {
                "mem_limit_bytes": mem_limit,
                "mem_limit_label": label,
                "hit_rate": hit_rate,
                "vanilla_predicted_s": vanilla_time,
                "ca_predicted_s": ca_time,
                "measured_trials_s": trial_times,
                "measured_mean_s": float(np.mean(trial_times)),
                "measured_min_s": float(np.min(trial_times)),
                "measured_max_s": float(np.max(trial_times)),
            }
        )

    # ------------------------------------------------------------------
    # Two-video reordering experiment
    # ------------------------------------------------------------------
    if video2 is not None:
        print("\n" + "=" * 60)
        print("Two-video reordering experiment")
        print(f"  Video A: {video} ({_format_bytes_human(file_size)})")
        print(f"  Video B: {video2} ({_format_bytes_human(file_size2)})")
        print("=" * 60)

        for mem_limit in mem_sweep:
            label = _format_bytes(mem_limit)

            # Predictions
            total_ab, _, _, _, _, _, _ = _predict_two_video_workflow_ca(
                float(file_size),
                float(file_size2),
                disk_bw,
                mem_bw,
                mem_limit,
                order="AB",
            )
            total_ba, _, _, _, _, _, _ = _predict_two_video_workflow_ca(
                float(file_size),
                float(file_size2),
                disk_bw,
                mem_bw,
                mem_limit,
                order="BA",
            )

            print(f"\nMemory limit: {label}")
            print(f"  CA pred A->B: {total_ab:.2f}s")
            print(f"  CA pred B->A: {total_ba:.2f}s")

            # Measurements: A->B
            ab_times: list[float] = []
            for trial in range(1, args.trials + 1):
                t = _run_two_video_sequential(
                    video,
                    video2,
                    out_dir,
                    mem_limit=mem_limit,
                    drop_caches=args.drop_caches,
                )
                ab_times.append(t)
                print(f"  A->B trial {trial}: {t:.2f}s")

            # Measurements: B->A
            ba_times: list[float] = []
            for trial in range(1, args.trials + 1):
                t = _run_two_video_sequential(
                    video2,
                    video,
                    out_dir,
                    mem_limit=mem_limit,
                    drop_caches=args.drop_caches,
                )
                ba_times.append(t)
                print(f"  B->A trial {trial}: {t:.2f}s")

            results["reorder"].append(
                {
                    "mem_limit_bytes": mem_limit,
                    "mem_limit_label": label,
                    "pred_ab_s": total_ab,
                    "pred_ba_s": total_ba,
                    "meas_ab_trials_s": ab_times,
                    "meas_ba_trials_s": ba_times,
                    "meas_ab_mean_s": float(np.mean(ab_times)),
                    "meas_ba_mean_s": float(np.mean(ba_times)),
                }
            )

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    out_json = out_dir / "exp1_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to: {out_json}")

    # ------------------------------------------------------------------
    # Plot: Figure-6 style (prediction vs measurement)
    # ------------------------------------------------------------------
    points = results["points"]
    mem_labels = [p["mem_limit_label"] for p in points]

    fig6_path = out_dir / "fig6_exp1.png"
    _plot_fig6(
        out_path=fig6_path,
        mem_labels=mem_labels,
        vanilla_pred=[p["vanilla_predicted_s"] for p in points],
        ca_pred=[p["ca_predicted_s"] for p in points],
        mean_s=[p["measured_mean_s"] for p in points],
        min_s=[p["measured_min_s"] for p in points],
        max_s=[p["measured_max_s"] for p in points],
        file_size_bytes=file_size,
    )
    print(f"Figure 6: {fig6_path}")

    # ------------------------------------------------------------------
    # Plot: Figure-7 style (2x2 bottleneck timeline + resource consumption)
    # ------------------------------------------------------------------
    cold_hit = points[0]["hit_rate"]
    warm_hit = points[-1]["hit_rate"]

    _, cold_progress, cold_bn, cold_sh = predict_cache_aware(
        float(file_size), disk_bw, mem_bw, cold_hit
    )
    _, warm_progress, warm_bn, warm_sh = predict_cache_aware(
        float(file_size), disk_bw, mem_bw, warm_hit
    )

    fig7_path = out_dir / "fig7_exp1_cold_vs_warm.png"
    _plot_fig7(
        out_path=fig7_path,
        cold_progress=cold_progress,
        cold_bottlenecks=cold_bn,
        cold_task=cold_sh,
        warm_progress=warm_progress,
        warm_bottlenecks=warm_bn,
        warm_task=warm_sh,
        disk_bw=disk_bw,
        mem_bw=mem_bw,
    )
    print(f"Figure 7: {fig7_path}")

    # ------------------------------------------------------------------
    # Plot: Figure-8 & 9 (two-video reordering)
    # ------------------------------------------------------------------
    if video2 is not None and results["reorder"]:
        reorder = results["reorder"]
        reorder_labels = [r["mem_limit_label"] for r in reorder]

        fig8_path = out_dir / "fig8_reordering.png"
        _plot_fig8_reordering(
            out_path=fig8_path,
            mem_labels=reorder_labels,
            pred_ab=[r["pred_ab_s"] for r in reorder],
            pred_ba=[r["pred_ba_s"] for r in reorder],
            meas_ab=[r["meas_ab_mean_s"] for r in reorder],
            meas_ba=[r["meas_ba_mean_s"] for r in reorder],
            file_size_a=file_size,
            file_size_b=file_size2,
        )
        print(f"Figure 8: {fig8_path}")

        # Fig 9: bottleneck detail for a memory limit where ordering matters
        # Pick the limit where |pred_ab - pred_ba| is largest
        diffs = [abs(r["pred_ab_s"] - r["pred_ba_s"]) for r in reorder]
        best_idx = int(np.argmax(diffs))
        best_mem = reorder[best_idx]["mem_limit_bytes"]

        total_ab, p1_ab, b1_ab, p2_ab, b2_ab, sh1_ab, sh2_ab = (
            _predict_two_video_workflow_ca(
                float(file_size),
                float(file_size2),
                disk_bw,
                mem_bw,
                best_mem,
                order="AB",
            )
        )
        total_ba, p1_ba, b1_ba, p2_ba, b2_ba, sh1_ba, sh2_ba = (
            _predict_two_video_workflow_ca(
                float(file_size),
                float(file_size2),
                disk_bw,
                mem_bw,
                best_mem,
                order="BA",
            )
        )

        if total_ab <= total_ba:
            best_label, worst_label = "A->B", "B->A"
            best_args = (p1_ab, b1_ab, sh1_ab, p2_ab, b2_ab, sh2_ab)
            worst_args = (p1_ba, b1_ba, sh1_ba, p2_ba, b2_ba, sh2_ba)
        else:
            best_label, worst_label = "B->A", "A->B"
            best_args = (p1_ba, b1_ba, sh1_ba, p2_ba, b2_ba, sh2_ba)
            worst_args = (p1_ab, b1_ab, sh1_ab, p2_ab, b2_ab, sh2_ab)

        fig9_path = out_dir / "fig9_bottleneck_reorder.png"
        _plot_fig9_bottleneck_reorder(
            out_path=fig9_path,
            prog1_best=best_args[0],
            bn1_best=best_args[1],
            sh1_best=best_args[2],
            prog2_best=best_args[3],
            bn2_best=best_args[4],
            sh2_best=best_args[5],
            prog1_worst=worst_args[0],
            bn1_worst=worst_args[1],
            sh1_worst=worst_args[2],
            prog2_worst=worst_args[3],
            bn2_worst=worst_args[4],
            sh2_worst=worst_args[5],
            best_label=best_label,
            worst_label=worst_label,
            disk_bw=disk_bw,
            mem_bw=mem_bw,
        )
        print(f"Figure 9: {fig9_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
