"""
Bottleneck Analysis Module

Provides functions for analyzing and interpreting bottleneck results.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List

from ..model.resources import ResourceType

if TYPE_CHECKING:  # Avoid import cycles during runtime
    from ..model.execution import TaskExecution


def compute_bottleneck_time(execution: "TaskExecution") -> Dict[ResourceType, float]:
    """
    Calculate time spent in each resource type bottleneck.

    Parameters
    ----------
    execution : TaskExecution
        Completed task execution with bottleneck data

    Returns
    -------
    Dict[ResourceType, float]
        Time spent bottlenecked on each resource type

    Examples
    --------
    >>> time_breakdown = compute_bottleneck_time(execution)
    >>> print(f"CPU bottleneck: {time_breakdown[ResourceType.CPU]:.2f}s")
    """
    time_by_type = defaultdict(float)

    if execution.progress_function is None or not execution.bottlenecks:
        return dict(time_by_type)

    for i, bottleneck in enumerate(execution.bottlenecks):
        if bottleneck is None:
            continue

        t_start = execution.progress_function.x[i]
        t_end = execution.progress_function.x[i + 1]
        duration = t_end - t_start

        # Get resource type
        if hasattr(bottleneck, "resource_type"):
            # ResourceRequirement
            time_by_type[bottleneck.resource_type] += duration
        else:
            # DataDependency - use special key
            time_by_type["DATA"] += duration

    return dict(time_by_type)


def compute_bottleneck_timeline_segments(execution: "TaskExecution") -> List[Dict]:
    """
    Get detailed timeline of bottleneck segments.

    Parameters
    ----------
    execution : TaskExecution
        Completed task execution

    Returns
    -------
    List[Dict]
        List of segments with start_time, end_time, duration, bottleneck

    Examples
    --------
    >>> segments = compute_bottleneck_timeline_segments(execution)
    >>> for seg in segments:
    ...     print(f"[{seg['start_time']:.2f}s - {seg['end_time']:.2f}s]: {seg['bottleneck']}")
    """
    segments = []

    if execution.progress_function is None or not execution.bottlenecks:
        return segments

    for i, bottleneck in enumerate(execution.bottlenecks):
        t_start = execution.progress_function.x[i]
        t_end = execution.progress_function.x[i + 1]

        segments.append(
            {
                "start_time": t_start,
                "end_time": t_end,
                "duration": t_end - t_start,
                "bottleneck": bottleneck,
                "bottleneck_name": str(bottleneck) if bottleneck else "Unknown",
            }
        )

    return segments


def print_bottleneck_timeline(execution: "TaskExecution"):
    """
    Print human-readable bottleneck timeline.

    Parameters
    ----------
    execution : TaskExecution
        Completed task execution

    Examples
    --------
    >>> print_bottleneck_timeline(execution)
    Bottleneck Timeline:
    ------------------------------------------------------------
    [   0.00s -  100.00s]: CPU_0 (CPU)
    [ 100.00s -  500.00s]: Cache (CACHE)
    [ 500.00s - 1200.00s]: Disk (DISK)
    ------------------------------------------------------------
    Total execution time: 1200.00s
    """
    print("Bottleneck Timeline:")
    print("-" * 60)

    if execution.progress_function is None or not execution.bottlenecks:
        print("  No bottleneck data available")
        print("-" * 60)
        return

    for i, bottleneck in enumerate(execution.bottlenecks):
        t_start = execution.progress_function.x[i]
        t_end = execution.progress_function.x[i + 1]

        if bottleneck is None:
            label = "Unknown"
        else:
            label = str(bottleneck)

        print(f"[{t_start:8.2f}s - {t_end:8.2f}s]: {label}")

    print("-" * 60)
    print(f"Total execution time: {execution.total_time():.2f}s")


def print_bottleneck_summary(execution: "TaskExecution"):
    """
    Print summary statistics of bottleneck time distribution.

    Parameters
    ----------
    execution : TaskExecution
        Completed task execution

    Examples
    --------
    >>> print_bottleneck_summary(execution)
    Bottleneck Time Distribution:
    ----------------------------------------
      CPU         :   300.00s ( 25.0%)
      CACHE       :   700.00s ( 58.3%)
      DISK        :   200.00s ( 16.7%)
    ----------------------------------------
    Total         :  1200.00s (100.0%)
    """
    time_breakdown = compute_bottleneck_time(execution)
    total_time = sum(time_breakdown.values())

    print("\nBottleneck Time Distribution:")
    print("-" * 40)

    if total_time == 0:
        print("  No execution time recorded")
        print("-" * 40)
        return

    # Sort by time (descending)
    sorted_items = sorted(time_breakdown.items(), key=lambda x: x[1], reverse=True)

    for resource_type, time in sorted_items:
        pct = 100 * time / total_time if total_time > 0 else 0
        if isinstance(resource_type, str):
            # Data dependency
            type_name = resource_type
        else:
            # ResourceType enum
            type_name = resource_type.name

        print(f"  {type_name:12s}: {time:8.2f}s ({pct:5.1f}%)")

    print("-" * 40)
    print(f"  {'Total':12s}: {total_time:8.2f}s (100.0%)")


def analyze_critical_path(execution: "TaskExecution") -> Dict:
    """
    Analyze the critical path of execution.

    Parameters
    ----------
    execution : TaskExecution
        Completed task execution

    Returns
    -------
    Dict
        Analysis results including:
        - total_time: Total execution time
        - critical_resource: Resource that dominated execution time
        - bottleneck_count: Number of distinct bottleneck segments
        - resource_breakdown: Time per resource type
    """
    time_breakdown = compute_bottleneck_time(execution)
    total_time = execution.total_time()

    if not time_breakdown:
        return {
            "total_time": total_time,
            "critical_resource": None,
            "bottleneck_count": 0,
            "resource_breakdown": {},
        }

    # Find critical resource (most time)
    critical_resource = max(time_breakdown.items(), key=lambda x: x[1])[0]

    return {
        "total_time": total_time,
        "critical_resource": critical_resource,
        "bottleneck_count": len(execution.bottlenecks),
        "resource_breakdown": time_breakdown,
    }


def compare_executions(
    executions: list["TaskExecution"], labels: list[str] | None = None
):
    """
    Compare multiple task executions.

    Parameters
    ----------
    executions : List[TaskExecution]
        List of executions to compare
    labels : List[str], optional
        Labels for each execution

    Examples
    --------
    >>> baseline = TaskExecution(req, env_baseline)
    >>> upgraded = TaskExecution(req, env_upgraded)
    >>> compare_executions([baseline, upgraded], ["Baseline", "With SSD"])
    """
    if labels is None:
        labels = [f"Execution {i + 1}" for i in range(len(executions))]

    print("\nExecution Comparison:")
    print("=" * 70)

    for execution_item, label in zip(executions, labels):
        print(f"\n{label}:")
        print("-" * 70)
        print(f"  Total time: {execution_item.total_time():.2f}s")

        time_breakdown = compute_bottleneck_time(execution_item)
        total_time = sum(time_breakdown.values())

        for resource_type, time in sorted(
            time_breakdown.items(), key=lambda x: x[1], reverse=True
        ):
            pct = 100 * time / total_time if total_time > 0 else 0
            if isinstance(resource_type, str):
                type_name = resource_type
            else:
                type_name = resource_type.name
            print(f"    {type_name:12s}: {time:7.2f}s ({pct:5.1f}%)")

    # Calculate speedups
    if len(executions) > 1:
        baseline_time = executions[0].total_time()
        print("\n" + "=" * 70)
        print("Speedup vs Baseline:")
        print("-" * 70)
        for i, (execution_item, label) in enumerate(zip(executions, labels)):
            if i == 0:
                print(f"  {label:30s}: 1.00x (baseline)")
            else:
                speedup = baseline_time / execution_item.total_time()
                print(f"  {label:30s}: {speedup:.2f}x")
