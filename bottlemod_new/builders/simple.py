"""
Simple Generic Task Builders

Provides helper functions to create generic linear tasks and environments.
These work for any resource type (CPU, memory, cache, disk, etc.).
"""

import math

from ..core import Func, PPoly
from ..model import ExecutionEnvironment, ResourceType, TaskRequirements


def create_linear_task(
    requirements: dict,
    start_progress: float = 0.0,
    end_progress: float = 1.0,
) -> TaskRequirements:
    """
    Create a task with multiple linear requirements.

    Parameters
    ----------
    requirements : dict
        Dictionary mapping (resource_type, name) tuples to requirement values
        Example: {
            (ResourceType.CPU, "CPU_0"): 1e12,
            (ResourceType.MEMORY, "RAM"): 100e9,
            (ResourceType.DISK, "SSD"): 50e9
        }
    start_progress : float, optional
        Starting progress value (default: 0.0)
    end_progress : float, optional
        Ending progress value (default: 1.0)

    Returns
    -------
    TaskRequirements
        Task with multiple linear requirements

    Examples
    --------
    >>> # Task with CPU and memory requirements
    >>> task = create_linear_task({
    ...     (ResourceType.CPU, "CPU_0"): 1e12,
    ...     (ResourceType.MEMORY, "RAM"): 100e9,
    ...     (ResourceType.DISK, "SSD"): 50e9
    ... })
    """
    task = TaskRequirements(max_progress=end_progress)

    for (resource_type, name), requirement in requirements.items():
        slope = requirement / (end_progress - start_progress)
        intercept = -slope * start_progress
        req_func = PPoly([start_progress, end_progress], [[slope], [intercept]])
        task.add_resource(resource_type, req_func, name)

    return task


def create_linear_environment(
    bandwidths: dict,
) -> ExecutionEnvironment:
    """
    Create an environment with multiple linear inputs.

    Parameters
    ----------
    bandwidths : dict
        Dictionary mapping (resource_type, name) tuples to bandwidth values
        Example: {
            (ResourceType.CPU, "CPU_0"): 50e9,
            (ResourceType.MEMORY, "RAM"): 20e9,
            (ResourceType.DISK, "SSD"): 2e9
        }

    Returns
    -------
    ExecutionEnvironment
        Environment with multiple linear inputs

    Examples
    --------
    >>> # Environment with CPU, memory, and disk
    >>> env = create_linear_environment({
    ...     (ResourceType.CPU, "CPU_0"): 50e9,
    ...     (ResourceType.MEMORY, "RAM"): 20e9,
    ...     (ResourceType.DISK, "SSD"): 2e9
    ... })
    """
    environment = ExecutionEnvironment()

    for (resource_type, name), bandwidth in bandwidths.items():
        input_func = PPoly([0, math.inf], [[bandwidth], [0]])
        environment.add_resource(resource_type, input_func, name)

    return environment


def create_task_with_data(
    cpu_requirement: float,
    data_requirement: float,
    cpu_name: str = "CPU_0",
    data_name: str = "Dataset",
    start_progress: float = 0.0,
    end_progress: float = 1.0,
) -> TaskRequirements:
    """
    Create a task with CPU requirement and data dependency.

    Parameters
    ----------
    cpu_requirement : float
        Total CPU work required (e.g., FLOPs)
    data_requirement : float
        Total data needed (e.g., bytes)
    cpu_name : str, optional
        Name for the CPU resource (default: "CPU_0")
    data_name : str, optional
        Name for the data dependency (default: "Dataset")
    start_progress : float, optional
        Starting progress value (default: 0.0)
    end_progress : float, optional
        Ending progress value (default: 1.0)

    Returns
    -------
    TaskRequirements
        Task with CPU requirement and data dependency

    Examples
    --------
    >>> # Task requiring 1 TFLOP and 100 GB of data
    >>> task = create_task_with_data(
    ...     cpu_requirement=1e12,
    ...     data_requirement=100e9,
    ...     data_name="Training_Data"
    ... )
    """
    # CPU requirement
    slope = cpu_requirement / (end_progress - start_progress)
    intercept = -slope * start_progress
    cpu_func = PPoly([start_progress, end_progress], [[slope], [intercept]])

    # Data dependency (progress unlocked per byte).
    # We expose the inverse requirement (bytes -> progress) so the core
    # algorithm can simply compose it with the data input function.
    if data_requirement <= 0:
        raise ValueError("data_requirement must be positive to define a dependency")

    data_progress_slope = (end_progress - start_progress) / data_requirement
    # Domain: [0 bytes, total bytes]. Beyond ``data_requirement`` the task has
    # already unlocked full progress, so keep a constant tail segment.
    tail_guard = data_requirement + 1.0
    data_func = Func(
        [0, data_requirement, tail_guard],
        [[data_progress_slope, 0.0], [start_progress, end_progress]],
    )

    return (
        TaskRequirements(max_progress=end_progress)
        .add_resource(ResourceType.CPU, cpu_func, cpu_name)
        .add_data(data_func, data_name)
    )


def create_environment_with_data(
    cpu_bandwidth: float,
    data_bandwidth: float,
    data_size: float,
    cpu_name: str = "CPU_0",
    data_name: str = "Dataset",
) -> ExecutionEnvironment:
    """
    Create an environment with CPU and data input.

    Parameters
    ----------
    cpu_bandwidth : float
        Available CPU bandwidth (e.g., FLOP/s)
    data_bandwidth : float
        Data transfer rate (e.g., bytes/s)
    data_size : float
        Total data size (e.g., bytes)
    cpu_name : str, optional
        Name for the CPU resource (default: "CPU_0")
    data_name : str, optional
        Name for the data input (default: "Dataset")

    Returns
    -------
    ExecutionEnvironment
        Environment with CPU and data inputs

    Examples
    --------
    >>> # Environment with 50 GFLOP/s CPU and 1 GB/s network
    >>> env = create_environment_with_data(
    ...     cpu_bandwidth=50e9,
    ...     data_bandwidth=1e9,
    ...     data_size=100e9,
    ...     data_name="Training_Data"
    ... )
    """
    # CPU input
    cpu_input = PPoly([0, math.inf], [[cpu_bandwidth], [0]])

    # Data input: linear until ``data_size`` is transferred, then constant.
    if data_size <= 0:
        raise ValueError("data_size must be positive to build a data input")
    if data_bandwidth <= 0:
        raise ValueError("data_bandwidth must be positive to build a data input")
    transfer_time = data_size / data_bandwidth
    tail_guard = transfer_time + 1.0
    data_input = Func(
        [0, transfer_time, tail_guard],
        [[data_bandwidth, 0.0], [0.0, data_size]],
    )

    return (
        ExecutionEnvironment()
        .add_resource(ResourceType.CPU, cpu_input, cpu_name)
        .add_data(data_input, data_name)
    )
