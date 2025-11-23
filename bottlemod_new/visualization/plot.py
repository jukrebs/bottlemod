"""
Visualization Module

Provides plotting utilities for bottleneck analysis results.
Inspired by the BottleMod paper (ICPE'25).
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..model import (
    ResourceInput,
    ResourceRequirement,
    ResourceType,
    TaskExecution,
)


def plot_bottleneck_analysis(
    execution: TaskExecution,
    time_points: int = 200,
    figsize: Tuple[float, float] = (12, 8),
    show_legend: bool = True,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot bottleneck analysis showing resource requirements over progress.

    Creates a visualization showing:
    - All resource requirement rates R'(p) over progress
    - All data dependency rates D'(p) over progress
    - Highlighted regions showing which resource is the bottleneck

    Parameters
    ----------
    execution : TaskExecution
        Executed task with bottleneck analysis results
    time_points : int, optional
        Number of points to sample for plotting (default: 200)
    figsize : Tuple[float, float], optional
        Figure size in inches (default: (12, 8))
    show_legend : bool, optional
        Whether to show legend (default: True)
    title : str, optional
        Plot title (default: auto-generated)

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    total_time = execution.total_time()
    time_arr = np.linspace(0, total_time, time_points)

    max_progress = execution.requirements.max_progress

    # Color schemes
    resource_colors = {
        ResourceType.CPU: "tab:blue",
        ResourceType.CACHE: "tab:orange",
        ResourceType.DISK: "tab:red",
        ResourceType.NETWORK: "tab:purple",
        ResourceType.MEMORY: "tab:green",
    }
    data_color = "tab:brown"

    # Track all lines for bottleneck highlighting
    resource_lines = []
    data_lines = []

    # Plot potential progress for each resource (dashed lines)
    for i, (req, inp) in enumerate(execution.resource_matches):
        # Get color and label
        color = resource_colors.get(req.resource_type, "tab:gray")
        label = req.name or f"{req.resource_type.name}_{i}"

        # Compute potential progress if only this resource mattered
        req_func = req.requirement_func
        inp_func = inp.input_func

        # Potential progress: solve R(p) = I(t) for p
        potential_progress_vals = []
        for t in time_arr:
            available = inp_func(t)
            # Find progress where requirement equals available input
            try:
                roots = req_func.solve(available)
                positive_roots = roots[roots >= 0]
                if len(positive_roots) > 0:
                    p = min(float(positive_roots[0]), max_progress)
                else:
                    p = 0
            except (ValueError, RuntimeError, TypeError):
                p = 0
            potential_progress_vals.append(p / max_progress * 100)

        (line,) = ax.plot(
            time_arr,
            potential_progress_vals,
            color=color,
            linewidth=1.5,
            linestyle="--",
            alpha=0.6,
            label=f"{label} (input)",
        )
        resource_lines.append((line, req))

    # Plot potential progress for data dependencies (dashed lines)
    for i, (dep, inp) in enumerate(execution.data_matches):
        label = dep.name or f"Data_{i}"

        dep_func = dep.dependency_func
        inp_func = inp.input_func

        # Potential progress from data
        potential_progress_vals = []
        for t in time_arr:
            available = inp_func(t)
            try:
                roots = dep_func.solve(available)
                positive_roots = roots[roots >= 0]
                if len(positive_roots) > 0:
                    p = min(float(positive_roots[0]), max_progress)
                else:
                    p = 0
            except (ValueError, RuntimeError, TypeError):
                p = 0
            potential_progress_vals.append(p / max_progress * 100)

        (line,) = ax.plot(
            time_arr,
            potential_progress_vals,
            color=data_color,
            linewidth=1.5,
            linestyle="--",
            alpha=0.6,
            label=f"{label} (input)",
        )
        data_lines.append((line, dep))

    # Plot actual progress (ONE solid line)
    actual_progress_vals = []
    for t in time_arr:
        p_actual = execution.progress_function(t)
        actual_progress_vals.append(p_actual / max_progress * 100)

    ax.plot(
        time_arr,
        actual_progress_vals,
        color="black",
        linewidth=2.5,
        label="Actual Progress",
        alpha=0.9,
    )

    # Highlight bottleneck regions
    _highlight_bottlenecks_time(ax, execution, resource_lines, data_lines)

    # Labels and formatting
    ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Progress (%)", fontsize=12, fontweight="bold")

    if title is None:
        title = f"Bottleneck Analysis (Total Time: {execution.total_time():.2f}s)"
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim(0, total_time)
    ax.set_ylim(0, 100)

    if show_legend:
        ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    return fig


def plot_progress_timeline(
    execution: TaskExecution,
    time_points: int = 200,
    figsize: Tuple[float, float] = (12, 6),
    show_bottlenecks: bool = True,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot progress P(t) over time with bottleneck annotations.

    Parameters
    ----------
    execution : TaskExecution
        Executed task with bottleneck analysis results
    time_points : int, optional
        Number of points to sample for plotting (default: 200)
    figsize : Tuple[float, float], optional
        Figure size in inches (default: (12, 6))
    show_bottlenecks : bool, optional
        Whether to annotate bottleneck transitions (default: True)
    title : str, optional
        Plot title (default: auto-generated)

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    total_time = execution.total_time()
    time_points_arr = np.linspace(0, total_time, time_points)

    # Evaluate progress function
    progress_values = np.array(
        [
            execution.progress_function(t) / execution.requirements.max_progress * 100
            for t in time_points_arr
        ]
    )

    ax.plot(
        time_points_arr,
        progress_values,
        color="tab:blue",
        linewidth=2.5,
        label="Progress P(t)",
    )

    # Mark bottleneck transitions
    if show_bottlenecks and execution.progress_function.x is not None:
        transition_times = execution.progress_function.x[1:-1]  # Skip endpoints
        for t in transition_times:
            if 0 < t < total_time:
                p = (
                    execution.progress_function(t)
                    / execution.requirements.max_progress
                    * 100
                )
                ax.axvline(t, color="red", linestyle=":", alpha=0.5, linewidth=1)
                ax.plot(t, p, "ro", markersize=6)

    # Labels
    ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Progress (%)", fontsize=12, fontweight="bold")

    if title is None:
        title = f"Progress Timeline (Total: {total_time:.2f}s)"
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim(0, total_time)
    ax.set_ylim(0, 105)
    ax.legend(loc="best")

    plt.tight_layout()
    return fig


def plot_combined_analysis(
    execution: TaskExecution,
    time_points: int = 200,
    figsize: Tuple[float, float] = (14, 10),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Create combined plot with both progress timeline and bottleneck analysis.

    Parameters
    ----------
    execution : TaskExecution
        Executed task with bottleneck analysis results
    time_points : int, optional
        Number of points to sample for plotting (default: 200)
    figsize : Tuple[float, float], optional
        Figure size in inches (default: (14, 10))
    title : str, optional
        Overall title (default: auto-generated)

    Returns
    -------
    plt.Figure
        The matplotlib figure with subplots
    """
    fig = plt.figure(figsize=figsize)

    if title is None:
        title = f"Task Execution Analysis (Total: {execution.total_time():.2f}s)"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Top: Progress over time
    ax1 = plt.subplot(2, 1, 1)
    _plot_progress_subplot(ax1, execution, time_points)

    # Bottom: Bottleneck analysis
    ax2 = plt.subplot(2, 1, 2)
    _plot_bottleneck_subplot(ax2, execution, time_points)

    plt.tight_layout()
    return fig


def _plot_progress_subplot(ax, execution: TaskExecution, time_points: int):
    """Helper to plot progress timeline in a subplot."""
    total_time = execution.total_time()
    time_arr = np.linspace(0, total_time, time_points)
    progress_vals = np.array(
        [
            execution.progress_function(t) / execution.requirements.max_progress * 100
            for t in time_arr
        ]
    )

    ax.plot(time_arr, progress_vals, color="tab:blue", linewidth=2.5)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Progress (%)", fontsize=11)
    ax.set_title("Progress P(t) over Time", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim(0, total_time)
    ax.set_ylim(0, 105)


def _plot_bottleneck_subplot(ax, execution: TaskExecution, time_points: int):
    """Plot instantaneous input/requirement rates over time."""
    total_time = execution.total_time()
    if total_time <= 0 or execution.progress_function is None:
        ax.set_visible(False)
        return

    time_arr = np.linspace(0, total_time, time_points)
    progress_func = execution.progress_function
    progress_rate_func = progress_func.derivative()
    progress_rates = np.array([float(progress_rate_func(t)) for t in time_arr])
    progress_rates = np.nan_to_num(progress_rates, nan=0.0, neginf=0.0, posinf=0.0)
    progress_rates[progress_rates < 0] = 0.0
    progress_vals = np.array([float(progress_func(t)) for t in time_arr])

    resource_colors = {
        ResourceType.CPU: "tab:blue",
        ResourceType.CACHE: "tab:orange",
        ResourceType.DISK: "tab:red",
        ResourceType.NETWORK: "tab:purple",
        ResourceType.MEMORY: "tab:green",
    }

    all_resource_matches: List[Tuple[ResourceRequirement, ResourceInput]] = []
    for matches in execution.resource_matches_by_type.values():
        all_resource_matches.extend(matches)

    for i, (req, inp) in enumerate(all_resource_matches):
        color = resource_colors.get(req.resource_type, "tab:gray")
        label = req.name or f"{req.resource_type.name}_{i}"

        req_func = req.requirement_func
        inp_func = inp.input_func

        input_rate_func = inp_func.derivative()
        requirement_rate_func = req_func.derivative()

        input_vals = []
        for t in time_arr:
            value = float(
                np.nan_to_num(input_rate_func(t), nan=0.0, neginf=0.0, posinf=0.0)
            )
            input_vals.append(max(0.0, value))

        requirement_vals = []
        for p_actual, progress_rate in zip(progress_vals, progress_rates):
            per_progress = float(
                np.nan_to_num(
                    requirement_rate_func(p_actual), nan=0.0, neginf=0.0, posinf=0.0
                )
            )
            requirement_vals.append(max(0.0, per_progress * progress_rate))

        ax.plot(
            time_arr,
            requirement_vals,
            color=color,
            linewidth=2,
            linestyle="-",
            alpha=0.9,
            label=f"{label} (required rate)",
        )

        ax.plot(
            time_arr,
            input_vals,
            color=color,
            linewidth=2,
            linestyle="--",
            alpha=0.7,
            label=f"{label} (input rate)",
        )

    # Plot input and requirement for data dependencies
    for i, (dep, inp) in enumerate(execution.data_matches):
        label = dep.name or f"Data_{i}"

        dep_func = dep.dependency_func
        inp_func = inp.input_func

        input_rate_func = inp_func.derivative()
        dependency_rate_func = dep_func.derivative()

        input_vals = []
        for t in time_arr:
            value = float(
                np.nan_to_num(input_rate_func(t), nan=0.0, neginf=0.0, posinf=0.0)
            )
            input_vals.append(max(0.0, value))

        requirement_vals = []
        for p_actual, progress_rate in zip(progress_vals, progress_rates):
            per_progress = float(
                np.nan_to_num(
                    dependency_rate_func(p_actual), nan=0.0, neginf=0.0, posinf=0.0
                )
            )
            requirement_vals.append(max(0.0, per_progress * progress_rate))

        ax.plot(
            time_arr,
            requirement_vals,
            color="tab:brown",
            linewidth=2,
            linestyle="-",
            alpha=0.9,
            label=f"{label} (required rate)",
        )

        ax.plot(
            time_arr,
            input_vals,
            color="tab:brown",
            linewidth=2,
            linestyle="--",
            alpha=0.7,
            label=f"{label} (input rate)",
        )

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Resource Rate", fontsize=11)
    ax.set_title("Input and Requirements Analysis", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim(0, total_time)
    ax.legend(loc="best", framealpha=0.9, fontsize=9)


def _highlight_bottlenecks_time(
    ax, execution: TaskExecution, resource_lines, data_lines
):
    """Highlight bottleneck regions with shading on time-based plot."""
    if execution.progress_function.x is None:
        return

    # Get time breakpoints
    time_breakpoints = execution.progress_function.x

    # For each segment, determine bottleneck and shade
    for i, bottleneck_obj in enumerate(execution.bottlenecks):
        if i >= len(time_breakpoints) - 1:
            break

        t_start = time_breakpoints[i]
        t_end = time_breakpoints[i + 1]

        # Find which line is the bottleneck
        bottleneck_color = None

        # Check resources
        for line, req in resource_lines:
            if bottleneck_obj == req:
                bottleneck_color = line.get_color()
                break

        # Check data dependencies
        if bottleneck_color is None:
            for line, dep in data_lines:
                if bottleneck_obj == dep:
                    bottleneck_color = line.get_color()
                    break

        # Add shaded region
        if bottleneck_color:
            ax.axvspan(t_start, t_end, alpha=0.15, color=bottleneck_color, zorder=0)
