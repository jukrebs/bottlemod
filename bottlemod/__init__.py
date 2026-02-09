"""Bottlemod package."""

from .func import Func
from .ppoly import PPoly
from .task import Task, TaskExecution
from .storage_hierarchy import (
    # Enums
    AccessType,
    ResourceType,
    # Process-side
    LogicalAccessProfile,
    # Environment-side
    StorageTier,
    TierMapping,
    # Cache behavior models
    CacheBehaviorModel,
    DirectHitRateModel,
    StackDistanceModel,
    PhaseBasedCacheModel,
    # Resource derivation
    derive_tier_resources,
    derive_all_tier_resources,
    # Convenience wrapper
    StorageHierarchyTask,
    # Utilities
    get_bottleneck_label,
    identify_bottleneck_type,
)

__all__ = [
    # Core BottleMod
    "Func",
    "PPoly", 
    "Task",
    "TaskExecution",
    # BottleMod-SH: Enums
    "AccessType",
    "ResourceType",
    # BottleMod-SH: Process-side
    "LogicalAccessProfile",
    # BottleMod-SH: Environment-side
    "StorageTier",
    "TierMapping",
    # BottleMod-SH: Cache behavior models
    "CacheBehaviorModel",
    "DirectHitRateModel",
    "StackDistanceModel",
    "PhaseBasedCacheModel",
    # BottleMod-SH: Resource derivation
    "derive_tier_resources",
    "derive_all_tier_resources",
    # BottleMod-SH: Convenience
    "StorageHierarchyTask",
    # BottleMod-SH: Utilities
    "get_bottleneck_label",
    "identify_bottleneck_type",
]
