"""
I/O-Aware Task Builders

Provides helper functions to create tasks with cache and disk I/O modeling.
"""

import math

from ..core import PPoly
from ..model import ExecutionEnvironment, ResourceType, TaskRequirements


def create_io_task(
    cpu_requirement: float,
    io_total_requirement: float,
    cache_hit_rate: float,
    cpu_name: str = "CPU_0",
    cache_name: str = "Cache",
    disk_name: str = "Disk",
    max_progress: float = 1.0,
) -> TaskRequirements:
    """
    Create task with CPU + I/O (cache/disk split).

    Automatically derives cache and disk requirements from:
    - Total I/O requirement
    - Cache hit rate

    Parameters
    ----------
    cpu_requirement : float
        Total CPU work required (e.g., FLOPs)
    io_total_requirement : float
        Total I/O bytes required
    cache_hit_rate : float
        Constant cache hit rate [0, 1]
    cpu_name : str, optional
        Name for CPU resource (default: "CPU_0")
    cache_name : str, optional
        Name for cache resource (default: "Cache")
    disk_name : str, optional
        Name for disk resource (default: "Disk")
    max_progress : float, optional
        Maximum progress value (default: 1.0)

    Returns
    -------
    TaskRequirements
        Task with CPU, cache, and disk requirements

    Examples
    --------
    >>> # Task: 1 TFLOP computation, 100 GB I/O, 80% cache hit rate
    >>> task = create_io_task(
    ...     cpu_requirement=1e12,
    ...     io_total_requirement=100e9,
    ...     cache_hit_rate=0.8
    ... )
    """
    # CPU requirement (linear)
    cpu_func = PPoly([0, max_progress], [[cpu_requirement], [0]])

    # Total I/O requirement (linear)
    io_total_func = PPoly([0, max_progress], [[io_total_requirement], [0]])

    # Constant hit rate
    hit_rate_func = PPoly([0, max_progress], [[0], [cache_hit_rate]])

    # Derive cache and disk requirements
    # R_cache(p) = ∫ h(p) * R'_IO(p) dp
    # R_disk(p) = ∫ (1-h(p)) * R'_IO(p) dp
    io_rate = io_total_func.derivative()
    cache_rate = io_rate * hit_rate_func
    cache_func = cache_rate.antiderivative()

    one_func = PPoly([0, max_progress], [[0], [1]])
    disk_rate = io_rate * (one_func - hit_rate_func)
    disk_func = disk_rate.antiderivative()

    return (
        TaskRequirements(max_progress=max_progress)
        .add_resource(ResourceType.CPU, cpu_func, cpu_name)
        .add_resource(ResourceType.CACHE, cache_func, cache_name)
        .add_resource(ResourceType.DISK, disk_func, disk_name)
    )


def create_io_environment(
    cpu_bandwidth: float,
    cache_bandwidth: float,
    disk_bandwidth: float,
    cpu_name: str = "CPU_0",
    cache_name: str = "Cache",
    disk_name: str = "Disk",
) -> ExecutionEnvironment:
    """
    Create environment with CPU + cache + disk.

    Parameters
    ----------
    cpu_bandwidth : float
        Available CPU bandwidth (e.g., FLOP/s)
    cache_bandwidth : float
        Cache bandwidth (e.g., bytes/s)
    disk_bandwidth : float
        Disk bandwidth (e.g., bytes/s)
    cpu_name : str, optional
        Name for CPU resource (default: "CPU_0")
    cache_name : str, optional
        Name for cache resource (default: "Cache")
    disk_name : str, optional
        Name for disk resource (default: "Disk")

    Returns
    -------
    ExecutionEnvironment
        Environment with CPU, cache, and disk inputs

    Examples
    --------
    >>> # Environment: 50 GFLOP/s CPU, 20 GB/s cache, 500 MB/s disk
    >>> env = create_io_environment(
    ...     cpu_bandwidth=50e9,
    ...     cache_bandwidth=20e9,
    ...     disk_bandwidth=500e6
    ... )
    """
    cpu_input = PPoly([0, math.inf], [[cpu_bandwidth], [0]])
    cache_input = PPoly([0, math.inf], [[cache_bandwidth], [0]])
    disk_input = PPoly([0, math.inf], [[disk_bandwidth], [0]])

    return (
        ExecutionEnvironment()
        .add_resource(ResourceType.CPU, cpu_input, cpu_name)
        .add_resource(ResourceType.CACHE, cache_input, cache_name)
        .add_resource(ResourceType.DISK, disk_input, disk_name)
    )


def create_warming_io_task(
    cpu_requirement: float,
    io_total_requirement: float,
    initial_hit_rate: float,
    final_hit_rate: float,
    cpu_name: str = "CPU_0",
    cache_name: str = "Cache",
    disk_name: str = "Disk",
    max_progress: float = 1.0,
) -> TaskRequirements:
    """
    Create task with cache warming up over time.

    Hit rate increases linearly from initial to final value,
    modeling a cold cache that gradually warms.

    Parameters
    ----------
    cpu_requirement : float
        Total CPU work required
    io_total_requirement : float
        Total I/O bytes required
    initial_hit_rate : float
        Cache hit rate at start (cold cache) [0, 1]
    final_hit_rate : float
        Cache hit rate at end (warm cache) [0, 1]
    cpu_name : str, optional
        Name for CPU resource (default: "CPU_0")
    cache_name : str, optional
        Name for cache resource (default: "Cache")
    disk_name : str, optional
        Name for disk resource (default: "Disk")
    max_progress : float, optional
        Maximum progress value (default: 1.0)

    Returns
    -------
    TaskRequirements
        Task with CPU, cache, and disk requirements (warming cache)

    Examples
    --------
    >>> # Task with cache warming from 10% to 90% hit rate
    >>> task = create_warming_io_task(
    ...     cpu_requirement=1e12,
    ...     io_total_requirement=100e9,
    ...     initial_hit_rate=0.1,
    ...     final_hit_rate=0.9
    ... )
    """
    # CPU requirement (linear)
    cpu_func = PPoly([0, max_progress], [[cpu_requirement], [0]])

    # Total I/O requirement (linear)
    io_total_func = PPoly([0, max_progress], [[io_total_requirement], [0]])

    # Linear hit rate ramp: h(p) = initial + (final - initial) * p
    slope = final_hit_rate - initial_hit_rate
    hit_rate_func = PPoly([0, max_progress], [[slope], [initial_hit_rate]])

    # Derive cache and disk requirements
    io_rate = io_total_func.derivative()
    cache_rate = io_rate * hit_rate_func
    cache_func = cache_rate.antiderivative()

    one_func = PPoly([0, max_progress], [[0], [1]])
    disk_rate = io_rate * (one_func - hit_rate_func)
    disk_func = disk_rate.antiderivative()

    return (
        TaskRequirements(max_progress=max_progress)
        .add_resource(ResourceType.CPU, cpu_func, cpu_name)
        .add_resource(ResourceType.CACHE, cache_func, cache_name)
        .add_resource(ResourceType.DISK, disk_func, disk_name)
    )


def create_custom_io_task(
    cpu_requirement: float,
    io_total_requirement: float,
    hit_rate_func: PPoly,
    cpu_name: str = "CPU_0",
    cache_name: str = "Cache",
    disk_name: str = "Disk",
    max_progress: float = 1.0,
) -> TaskRequirements:
    """
    Create task with custom (non-linear) cache hit rate function.

    Allows full control over hit rate behavior h(p).

    Parameters
    ----------
    cpu_requirement : float
        Total CPU work required
    io_total_requirement : float
        Total I/O bytes required
    hit_rate_func : PPoly
        Custom hit rate function h(p) over progress
    cpu_name : str, optional
        Name for CPU resource (default: "CPU_0")
    cache_name : str, optional
        Name for cache resource (default: "Cache")
    disk_name : str, optional
        Name for disk resource (default: "Disk")
    max_progress : float, optional
        Maximum progress value (default: 1.0)

    Returns
    -------
    TaskRequirements
        Task with CPU, cache, and disk requirements (custom hit rate)

    Examples
    --------
    >>> # Piecewise hit rate: 70% → 30% → 80%
    >>> hit_rate = PPoly([0, 0.2, 0.8, 1.0],
    ...                  [[0, 0.70], [-0.667, 0.83], [0, 0.80]])
    >>> task = create_custom_io_task(
    ...     cpu_requirement=1e12,
    ...     io_total_requirement=100e9,
    ...     hit_rate_func=hit_rate
    ... )
    """
    # CPU requirement (linear)
    cpu_func = PPoly([0, max_progress], [[cpu_requirement], [0]])

    # Total I/O requirement (linear)
    io_total_func = PPoly([0, max_progress], [[io_total_requirement], [0]])

    # Derive cache and disk requirements
    io_rate = io_total_func.derivative()
    cache_rate = io_rate * hit_rate_func
    cache_func = cache_rate.antiderivative()

    one_func = PPoly([0, max_progress], [[0], [1]])
    disk_rate = io_rate * (one_func - hit_rate_func)
    disk_func = disk_rate.antiderivative()

    return (
        TaskRequirements(max_progress=max_progress)
        .add_resource(ResourceType.CPU, cpu_func, cpu_name)
        .add_resource(ResourceType.CACHE, cache_func, cache_name)
        .add_resource(ResourceType.DISK, disk_func, disk_name)
    )


def create_multi_level_io_task(
    cpu_requirement: float,
    io_requirements: dict,
    cpu_name: str = "CPU_0",
    max_progress: float = 1.0,
) -> TaskRequirements:
    """
    Create task with multiple levels of I/O (e.g., L1, L2, L3, RAM, Disk).

    Parameters
    ----------
    cpu_requirement : float
        Total CPU work required
    io_requirements : dict
        Dictionary mapping (resource_type, name, hit_rate) to I/O requirement
        Example: {
            (ResourceType.CACHE, "L1", 0.9): 10e9,
            (ResourceType.CACHE, "L2", 0.08): 10e9,
            (ResourceType.DISK, "SSD", 0.02): 10e9
        }
    cpu_name : str, optional
        Name for CPU resource (default: "CPU_0")
    max_progress : float, optional
        Maximum progress value (default: 1.0)

    Returns
    -------
    TaskRequirements
        Task with CPU and multiple I/O levels

    Examples
    --------
    >>> # Multi-level cache hierarchy
    >>> task = create_multi_level_io_task(
    ...     cpu_requirement=1e12,
    ...     io_requirements={
    ...         (ResourceType.CACHE, "L1", 0.80): 100e9,
    ...         (ResourceType.CACHE, "L2", 0.15): 100e9,
    ...         (ResourceType.MEMORY, "RAM", 0.04): 100e9,
    ...         (ResourceType.DISK, "SSD", 0.01): 100e9
    ...     }
    ... )
    """
    # CPU requirement
    cpu_func = PPoly([0, max_progress], [[cpu_requirement], [0]])

    task = TaskRequirements(max_progress=max_progress).add_resource(
        ResourceType.CPU, cpu_func, cpu_name
    )

    # Add each I/O level
    for (resource_type, name, hit_rate), io_req in io_requirements.items():
        # Each level gets its hit_rate fraction of total I/O
        io_func = PPoly([0, max_progress], [[io_req], [0]])
        hit_func = PPoly([0, max_progress], [[0], [hit_rate]])

        io_rate = io_func.derivative()
        level_rate = io_rate * hit_func
        level_func = level_rate.antiderivative()

        task.add_resource(resource_type, level_func, name)

    return task


def create_multi_level_io_environment(
    cpu_bandwidth: float,
    io_bandwidths: dict,
    cpu_name: str = "CPU_0",
) -> ExecutionEnvironment:
    """
    Create environment with multiple I/O levels.

    Parameters
    ----------
    cpu_bandwidth : float
        Available CPU bandwidth
    io_bandwidths : dict
        Dictionary mapping (resource_type, name) to bandwidth
        Example: {
            (ResourceType.CACHE, "L1"): 100e9,
            (ResourceType.CACHE, "L2"): 50e9,
            (ResourceType.MEMORY, "RAM"): 20e9,
            (ResourceType.DISK, "SSD"): 2e9
        }
    cpu_name : str, optional
        Name for CPU resource (default: "CPU_0")

    Returns
    -------
    ExecutionEnvironment
        Environment with CPU and multiple I/O levels

    Examples
    --------
    >>> # Multi-level cache hierarchy environment
    >>> env = create_multi_level_io_environment(
    ...     cpu_bandwidth=50e9,
    ...     io_bandwidths={
    ...         (ResourceType.CACHE, "L1"): 100e9,
    ...         (ResourceType.CACHE, "L2"): 50e9,
    ...         (ResourceType.MEMORY, "RAM"): 20e9,
    ...         (ResourceType.DISK, "SSD"): 2e9
    ...     }
    ... )
    """
    cpu_input = PPoly([0, math.inf], [[cpu_bandwidth], [0]])

    env = ExecutionEnvironment().add_resource(ResourceType.CPU, cpu_input, cpu_name)

    for (resource_type, name), bandwidth in io_bandwidths.items():
        io_input = PPoly([0, math.inf], [[bandwidth], [0]])
        env.add_resource(resource_type, io_input, name)

    return env
