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
    WSSModel,
    LRUEvictionModel,
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
    # BottleMod-CA: Enums
    "AccessType",
    "ResourceType",
    # BottleMod-CA: Process-side
    "LogicalAccessProfile",
    # BottleMod-CA: Environment-side
    "StorageTier",
    "TierMapping",
    # BottleMod-CA: Cache behavior models
    "CacheBehaviorModel",
    "DirectHitRateModel",
    "WSSModel",
    "LRUEvictionModel",
    # BottleMod-CA: Resource derivation
    "derive_tier_resources",
    "derive_all_tier_resources",
    # BottleMod-CA: Convenience
    "StorageHierarchyTask",
    # BottleMod-CA: Utilities
    "get_bottleneck_label",
    "identify_bottleneck_type",
]
