"""
Write Cache Builders

Provides helper functions to create tasks with write-back cache modeling.

Linux uses write-back caching for writes:
- Writes go to page cache (memory) at cache speed
- Dirty pages are flushed to disk asynchronously
- Application sees cache speed until dirty limit is reached
- After dirty limit, writes are throttled to disk speed

This module models this two-phase behavior:
1. Burst phase: writes at cache bandwidth (dirty buffer filling)
2. Disk-limited phase: writes at disk bandwidth (dirty limit reached)
"""

import math
from typing import Optional

from ..core import PPoly
from ..model import ExecutionEnvironment, ResourceType, TaskRequirements


def create_write_task(
    write_total: float,
    dirty_limit: float,
    cpu_requirement: Optional[float] = None,
    cpu_name: str = "CPU_0",
    disk_name: str = "Disk",
    max_progress: float = 1.0,
) -> TaskRequirements:
    """
    Create task with write-back cache modeling.

    Models the two-phase write behavior:
    1. Burst phase (progress 0 to p_transition):
       - All writes go to cache at cache speed
       - Dirty buffer fills up
    2. Disk-limited phase (progress p_transition to 1):
       - Dirty limit reached, writes throttled to disk speed
       - Background flushing keeps dirty pages at limit

    The transition point p_transition = min(dirty_limit / write_total, 1.0)

    Parameters
    ----------
    write_total : float
        Total bytes to write
    dirty_limit : float
        Maximum dirty pages before writes are throttled (bytes)
        Corresponds to Linux dirty_bytes or dirty_ratio * memory
    cpu_requirement : float, optional
        Total CPU work required (e.g., FLOPs). If None, no CPU requirement.
    cpu_name : str, optional
        Name for CPU resource (default: "CPU_0")
    cache_name : str, optional
        Name for cache resource (default: "WriteCache")
    disk_name : str, optional
        Name for disk resource (default: "Disk")
    max_progress : float, optional
        Maximum progress value (default: 1.0)

    Returns
    -------
    TaskRequirements
        Task with cache and disk requirements modeling write-back caching

    Examples
    --------
    >>> # Small write that fits in cache (10 GB write, 16 GB dirty limit)
    >>> task = create_write_task(
    ...     write_total=10e9,
    ...     dirty_limit=16e9
    ... )
    >>> # All writes go to cache - no disk requirement

    >>> # Large write exceeding cache (100 GB write, 16 GB dirty limit)
    >>> task = create_write_task(
    ...     write_total=100e9,
    ...     dirty_limit=16e9
    ... )
    >>> # First 16% at cache speed, remaining 84% at disk speed

    Notes
    -----
    The model assumes:
    - Writes are sequential (no random access overhead)
    - Background flush keeps pace with incoming writes after dirty limit
    - No I/O scheduler delays or other system overheads
    """
    task = TaskRequirements(max_progress=max_progress)

    # Add a "dummy" data dependency that doesn't constrain progress
    # This is needed because the algorithm uses data progress as the baseline
    # Data dependency: 1 data unit = 1 progress unit
    data_dep = PPoly([0, max_progress], [[1], [0]])
    task.add_data(dependency_func=data_dep, name="WriteData")

    # Add CPU requirement if specified
    if cpu_requirement is not None:
        cpu_func = PPoly([0, max_progress], [[cpu_requirement], [0]])
        task.add_resource(ResourceType.CPU, cpu_func, cpu_name)

    # Validate parameters
    if write_total <= 0:
        raise ValueError("write_total must be positive")
    if dirty_limit < 0:
        raise ValueError("dirty_limit must be non-negative")

    # Model: Single write resource with total write_total bytes required
    # The write goes to "effective storage" - cache or disk depending on phase
    # The environment builder will set up appropriate bandwidth transitions
    write_func = PPoly([0, max_progress], [[write_total], [0]])
    task.add_resource(ResourceType.DISK, write_func, disk_name)

    return task


def create_write_environment(
    cache_bandwidth: float,
    disk_bandwidth: Optional[float] = None,
    dirty_limit: Optional[float] = None,
    cpu_bandwidth: Optional[float] = None,
    cpu_name: str = "CPU_0",
    disk_name: str = "Disk",
) -> ExecutionEnvironment:
    """
    Create environment for write-back cache modeling.

    For large writes that exceed the dirty limit, this creates a piecewise
    bandwidth function:
    - Phase 1 (0 to t_transition): Write at cache_bandwidth
    - Phase 2 (t_transition to inf): Write at disk_bandwidth

    Where t_transition = dirty_limit / cache_bandwidth

    Parameters
    ----------
    cache_bandwidth : float
        Cache/memory write bandwidth (bytes/s)
        Typical values: DDR4 ~25 GB/s, DDR5 ~50 GB/s
    disk_bandwidth : float, optional
        Disk write bandwidth (bytes/s). If None, assumes infinite disk bandwidth
        (use this when all writes fit in cache).
        Typical values: HDD ~150 MB/s, SATA SSD ~500 MB/s, NVMe ~3 GB/s
    dirty_limit : float, optional
        Dirty page limit in bytes. Used to compute when bandwidth transitions
        from cache to disk. If None, no transition occurs (cache-only mode).
    cpu_bandwidth : float, optional
        CPU bandwidth (e.g., FLOP/s). If None, no CPU input.
    cpu_name : str, optional
        Name for CPU resource (default: "CPU_0")
    disk_name : str, optional
        Name for disk resource (default: "Disk")

    Returns
    -------
    ExecutionEnvironment
        Environment with write bandwidth (piecewise if dirty_limit specified)

    Examples
    --------
    >>> # Small write that fits in cache - cache speed only
    >>> env = create_write_environment(cache_bandwidth=25e9)

    >>> # Large write - transitions from cache to disk at dirty_limit
    >>> env = create_write_environment(
    ...     cache_bandwidth=25e9,   # 25 GB/s
    ...     disk_bandwidth=500e6,   # 500 MB/s
    ...     dirty_limit=16e9        # 16 GB dirty limit
    ... )
    """
    env = ExecutionEnvironment()

    # Add data input matching the task's data dependency
    # Data instantly available (constant max_progress), doesn't constrain progress
    data_input = PPoly([0, math.inf], [[0], [1.0]])  # Constant 1.0 (max_progress)
    env.add_data(input_func=data_input, name="WriteData")

    if cpu_bandwidth is not None:
        cpu_input = PPoly([0, math.inf], [[cpu_bandwidth], [0]])
        env.add_resource(ResourceType.CPU, cpu_input, cpu_name)

    # Create write bandwidth function
    if dirty_limit is not None and disk_bandwidth is not None:
        # Two-phase bandwidth: cache speed until dirty_limit, then disk speed
        # Time when dirty_limit is reached: t_transition = dirty_limit / cache_bandwidth
        t_transition = dirty_limit / cache_bandwidth

        # Cumulative bytes at transition (from cache phase)
        bytes_at_transition = dirty_limit  # = cache_bandwidth * t_transition

        # Piecewise cumulative function:
        # Phase 1: I(t) = cache_bandwidth * t  for t in [0, t_transition]
        # Phase 2: I(t) = dirty_limit + disk_bandwidth * (t - t_transition)
        #               = disk_bandwidth * t + (dirty_limit - disk_bandwidth * t_transition)
        intercept_2 = bytes_at_transition - disk_bandwidth * t_transition

        write_input = PPoly(
            [0, t_transition, math.inf],
            [[cache_bandwidth, disk_bandwidth], [0, intercept_2]],
        )
    else:
        # Single-phase: constant cache bandwidth
        write_input = PPoly([0, math.inf], [[cache_bandwidth], [0]])

    env.add_resource(ResourceType.DISK, write_input, disk_name)

    return env


def create_write_task_with_flush(
    write_total: float,
    dirty_limit: float,
    dirty_background_limit: float,
    cpu_requirement: Optional[float] = None,
    cpu_name: str = "CPU_0",
    disk_name: str = "Disk",
    max_progress: float = 1.0,
) -> TaskRequirements:
    """
    Create task with more realistic write-back cache modeling including
    background flush behavior.

    Linux has two dirty thresholds:
    1. dirty_background_bytes: When background flushing starts
    2. dirty_bytes: When writes start blocking

    This models three phases:
    1. Pure cache phase (0 to p_bg): All writes to cache, no disk I/O
    2. Mixed phase (p_bg to p_dirty): Cache writes + background disk flush
    3. Throttled phase (p_dirty to 1): Disk-limited writes

    Parameters
    ----------
    write_total : float
        Total bytes to write
    dirty_limit : float
        Maximum dirty pages before writes block (dirty_bytes)
    dirty_background_limit : float
        Dirty page threshold when background flush starts (dirty_background_bytes)
        Must be <= dirty_limit
    cpu_requirement : float, optional
        Total CPU work required. If None, no CPU requirement.
    cpu_name : str, optional
        Name for CPU resource (default: "CPU_0")
    cache_name : str, optional
        Name for cache resource (default: "WriteCache")
    disk_name : str, optional
        Name for disk resource (default: "Disk")
    max_progress : float, optional
        Maximum progress value (default: 1.0)

    Returns
    -------
    TaskRequirements
        Task with cache and disk requirements

    Examples
    --------
    >>> # Typical Linux defaults: 10% background, 20% hard limit
    >>> # Assuming 32 GB RAM: 3.2 GB background, 6.4 GB hard limit
    >>> task = create_write_task_with_flush(
    ...     write_total=50e9,
    ...     dirty_limit=6.4e9,
    ...     dirty_background_limit=3.2e9
    ... )
    """
    if dirty_background_limit > dirty_limit:
        raise ValueError("dirty_background_limit must be <= dirty_limit")

    task = TaskRequirements(max_progress=max_progress)

    # Add a "dummy" data dependency that doesn't constrain progress
    data_dep = PPoly([0, max_progress], [[1], [0]])
    task.add_data(dependency_func=data_dep, name="WriteData")

    # Add CPU requirement if specified
    if cpu_requirement is not None:
        cpu_func = PPoly([0, max_progress], [[cpu_requirement], [0]])
        task.add_resource(ResourceType.CPU, cpu_func, cpu_name)

    if write_total <= 0:
        raise ValueError("write_total must be positive")

    # Simplified model: Use single write resource with total write_total bytes
    # The environment will handle the bandwidth transitions based on dirty_limit
    # and dirty_background_limit (passed when creating environment)
    write_func = PPoly([0, max_progress], [[write_total], [0]])
    task.add_resource(ResourceType.DISK, write_func, disk_name)

    return task


def create_sync_write_task(
    write_total: float,
    cpu_requirement: Optional[float] = None,
    cpu_name: str = "CPU_0",
    disk_name: str = "Disk",
    max_progress: float = 1.0,
) -> TaskRequirements:
    """
    Create task for synchronous (O_SYNC/O_DIRECT) writes.

    Synchronous writes bypass the page cache entirely and write directly
    to disk. This models applications using:
    - O_SYNC flag (synchronous I/O)
    - O_DIRECT flag (direct I/O, bypassing cache)
    - fsync() after each write

    Parameters
    ----------
    write_total : float
        Total bytes to write
    cpu_requirement : float, optional
        Total CPU work required. If None, no CPU requirement.
    cpu_name : str, optional
        Name for CPU resource (default: "CPU_0")
    disk_name : str, optional
        Name for disk resource (default: "Disk")
    max_progress : float, optional
        Maximum progress value (default: 1.0)

    Returns
    -------
    TaskRequirements
        Task with only disk requirement (no cache benefit)

    Examples
    --------
    >>> # Database transaction log (typically O_SYNC)
    >>> task = create_sync_write_task(write_total=1e9)
    """
    task = TaskRequirements(max_progress=max_progress)

    # Add a "dummy" data dependency that doesn't constrain progress
    data_dep = PPoly([0, max_progress], [[1], [0]])
    task.add_data(dependency_func=data_dep, name="WriteData")

    if cpu_requirement is not None:
        cpu_func = PPoly([0, max_progress], [[cpu_requirement], [0]])
        task.add_resource(ResourceType.CPU, cpu_func, cpu_name)

    # All writes go directly to disk
    disk_func = PPoly([0, max_progress], [[write_total], [0]])
    task.add_resource(ResourceType.DISK, disk_func, disk_name)

    return task


def create_sync_write_environment(
    disk_bandwidth: float,
    cpu_bandwidth: Optional[float] = None,
    cpu_name: str = "CPU_0",
    disk_name: str = "Disk",
) -> ExecutionEnvironment:
    """
    Create environment for synchronous write tasks.

    Parameters
    ----------
    disk_bandwidth : float
        Disk write bandwidth (bytes/s)
    cpu_bandwidth : float, optional
        CPU bandwidth. If None, no CPU input.
    cpu_name : str, optional
        Name for CPU resource (default: "CPU_0")
    disk_name : str, optional
        Name for disk resource (default: "Disk")

    Returns
    -------
    ExecutionEnvironment
        Environment with disk input only

    Examples
    --------
    >>> env = create_sync_write_environment(disk_bandwidth=500e6)
    """
    env = ExecutionEnvironment()

    # Add data input matching the task's data dependency
    data_input = PPoly([0, math.inf], [[0], [1.0]])
    env.add_data(input_func=data_input, name="WriteData")

    if cpu_bandwidth is not None:
        cpu_input = PPoly([0, math.inf], [[cpu_bandwidth], [0]])
        env.add_resource(ResourceType.CPU, cpu_input, cpu_name)

    disk_input = PPoly([0, math.inf], [[disk_bandwidth], [0]])
    env.add_resource(ResourceType.DISK, disk_input, disk_name)

    return env
