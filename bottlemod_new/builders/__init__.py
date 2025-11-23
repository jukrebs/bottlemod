"""
BottleMod Builders Module

This module provides helper functions to easily create common task scenarios:
- CPU-specific tasks and environments
- I/O-aware tasks (cache/disk modeling)
- Generic linear tasks for any resource type
- Unit conversion helpers (GFLOPS, GB, GBps, percent, etc.)
"""

# CPU-specific builders
from .cpu import (
    create_cpu_environment,
    create_cpu_task,
)

# Unit conversion helpers
from .helper import (
    GB,
    GFLOPS,
    # Bytes
    KB,
    # FLOPS
    KFLOPS,
    MB,
    MFLOPS,
    PB,
    PFLOPS,
    TB,
    TFLOPS,
    GBps,
    GHz,
    # Bandwidth
    KBps,
    # Frequency
    KHz,
    MBps,
    MHz,
    TBps,
    hit_rate,
    hours,
    minutes,
    miss_rate,
    # Time
    ms,
    ns,
    # Patterns
    percent,
    us,
)

# I/O-aware builders
from .io import (
    create_custom_io_task,
    create_io_environment,
    create_io_task,
    create_multi_level_io_environment,
    create_multi_level_io_task,
    create_warming_io_task,
)

# Generic linear builders
from .simple import (
    create_environment_with_data,
    create_linear_environment,
    create_linear_task,
    create_task_with_data,
)

__all__ = [
    # CPU builders
    "create_cpu_task",
    "create_cpu_environment",
    # I/O builders
    "create_io_task",
    "create_io_environment",
    "create_warming_io_task",
    "create_custom_io_task",
    "create_multi_level_io_task",
    "create_multi_level_io_environment",
    # Generic builders
    "create_linear_task",
    "create_linear_environment",
    "create_task_with_data",
    "create_environment_with_data",
    # Helper functions
    "KFLOPS",
    "MFLOPS",
    "GFLOPS",
    "TFLOPS",
    "PFLOPS",
    "KB",
    "MB",
    "GB",
    "TB",
    "PB",
    "KBps",
    "MBps",
    "GBps",
    "TBps",
    "ms",
    "us",
    "ns",
    "minutes",
    "hours",
    "KHz",
    "MHz",
    "GHz",
    "percent",
    "hit_rate",
    "miss_rate",
]
