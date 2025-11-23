"""
BottleMod Model Module

This module provides the main modeling components for BottleMod:
- TaskRequirements: What a task needs (requirements)
- ExecutionEnvironment: What hardware provides (inputs)
- TaskExecution: Bottleneck analysis results
"""

from .environment import ExecutionEnvironment
from .execution import TaskExecution
from .requirements import TaskRequirements
from .resources import (
    DataDependency,
    DataInput,
    Input,
    Resource,
    ResourceInput,
    ResourceRequirement,
    ResourceType,
)

__all__ = [
    "TaskRequirements",
    "ExecutionEnvironment",
    "TaskExecution",
    # Resource types
    "ResourceType",
    "ResourceRequirement",
    "ResourceInput",
    "DataDependency",
    "DataInput",
    # Aliases
    "Resource",
    "Input",
]
