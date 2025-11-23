"""
CPU Task Builders

Provides helper functions to create CPU-specific task scenarios.
"""

import math

from ..core import PPoly
from ..model import ExecutionEnvironment, ResourceType, TaskRequirements


def create_cpu_task(
    cpu_requirement: float,
    cpu_name: str = "CPU_0",
    start_progress: float = 0.0,
    end_progress: float = 1.0,
) -> TaskRequirements:
    """
    Create a simple CPU-only task.

    Parameters
    ----------
    cpu_requirement : float
        Total CPU work required (e.g., FLOPs)
    cpu_name : str, optional
        Name for the CPU resource (default: "CPU_0")
    start_progress : float, optional
        Starting progress value (default: 0.0)
    end_progress : float, optional
        Ending progress value (default: 1.0)

    Returns
    -------
    TaskRequirements
        Task with single CPU requirement

    """
    slope = cpu_requirement / (end_progress - start_progress)
    intercept = -slope * start_progress
    cpu_func = PPoly([start_progress, end_progress], [[slope], [intercept]])

    return TaskRequirements(max_progress=end_progress).add_resource(
        ResourceType.CPU, cpu_func, cpu_name
    )


def create_cpu_environment(
    cpu_bandwidth: float,
    cpu_name: str = "CPU_0",
) -> ExecutionEnvironment:
    """
    Create a simple CPU-only execution environment.

    Parameters
    ----------
    cpu_bandwidth : float
        Available CPU bandwidth (e.g., FLOP/s)
    cpu_name : str, optional
        Name for the CPU resource (default: "CPU_0")

    Returns
    -------
    ExecutionEnvironment
        Environment with single CPU input
    """
    cpu_input = PPoly([0, math.inf], [[cpu_bandwidth], [0]])

    return ExecutionEnvironment().add_resource(ResourceType.CPU, cpu_input, cpu_name)
