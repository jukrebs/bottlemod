"""
Resource type definitions for BottleMod.

This module defines resource types, requirements, and inputs used in
bottleneck analysis.
"""

from enum import Enum, auto
from typing import NamedTuple

from ..core.func import Func
from ..core.ppoly import PPoly


class ResourceType(Enum):
    """Enumeration of resource types that can bottleneck task execution."""

    CPU = auto()
    MEMORY = auto()
    CACHE = auto()
    DISK = auto()
    NETWORK = auto()
    GPU = auto()
    CUSTOM = auto()

    def __str__(self):
        return self.name


class ResourceRequirement(NamedTuple):
    """
    A single resource requirement for a task.

    Attributes
    ----------
    resource_type : ResourceType
        Type of resource (CPU, CACHE, DISK, etc.)
    requirement_func : PPoly
        Cumulative requirement function R(p) mapping progress -> total resource needed
    name : str
        Optional human-readable name (e.g., "CPU_0", "NVMe_SSD")
    """

    resource_type: ResourceType
    requirement_func: PPoly
    name: str = ""

    def __str__(self):
        if self.name:
            return f"{self.name} ({self.resource_type.name})"
        return self.resource_type.name

    def __repr__(self):
        name_part = f", name='{self.name}'" if self.name else ""
        return f"ResourceRequirement({self.resource_type}{name_part})"


class ResourceInput(NamedTuple):
    """
    A single resource input (available capacity) from the execution environment.

    Attributes
    ----------
    resource_type : ResourceType
        Type of resource being provided
    input_func : PPoly
        Cumulative input function I(t) mapping time -> total resource delivered
    name : str
        Optional human-readable name (e.g., "CPU_0", "Cache_DDR4")
    """

    resource_type: ResourceType
    input_func: PPoly
    name: str = ""

    def __str__(self):
        if self.name:
            return f"{self.name} ({self.resource_type.name})"
        return self.resource_type.name

    def __repr__(self):
        name_part = f", name='{self.name}'" if self.name else ""
        return f"ResourceInput({self.resource_type}{name_part})"


class DataDependency(NamedTuple):
    """
    A data availability dependency for a task.

    Unlike resources (which limit speed), data dependencies cap the maximum
    achievable progress given how much data has arrived.  To keep the math
    close to the paper (§3.1), ``dependency_func`` is the inverse of the
    cumulative data requirement: it maps *available data* (e.g. bytes) to the
    *maximum progress* that can be unlocked (monotonically increasing and
    bounded by the task's max progress).

    Attributes
    ----------
    dependency_func : Func
        Monotonic function R^{-1}_{D,k}(n) returning achievable progress for
        ``n`` units of data
    name : str
        Optional human-readable name (e.g., "Dataset_A", "Model_Weights")
    """

    dependency_func: Func
    name: str = ""

    def __str__(self):
        return f"Data: {self.name}" if self.name else "Data Dependency"

    def __repr__(self):
        name_part = f", name='{self.name}'" if self.name else ""
        return f"DataDependency({name_part.lstrip(', ')})"


class DataInput(NamedTuple):
    """
    Data availability over time from the execution environment.

    Attributes
    ----------
    input_func : Func
        Cumulative data availability I_D(t) mapping time -> total data available
    name : str
        Optional human-readable name matching the dependency
    """

    input_func: Func
    name: str = ""

    def __str__(self):
        return f"Data Input: {self.name}" if self.name else "Data Input"

    def __repr__(self):
        name_part = f", name='{self.name}'" if self.name else ""
        return f"DataInput({name_part.lstrip(', ')})"


# Type aliases for convenience
Resource = ResourceRequirement
Input = ResourceInput
