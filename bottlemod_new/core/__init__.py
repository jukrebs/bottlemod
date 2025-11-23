"""
BottleMod Core Module

This module provides the foundational components for BottleMod:
- Piecewise polynomial functions (PPoly)
- Monotonic functions (Func)
- Resource type definitions
- Core bottleneck analysis algorithms
"""

from .algorithm import (
    bottleneck_analysis,
    get_speed,
    last_segment,
    next_segment_only,
    ppoly_min,
    ppoly_min2,
    ppoly_min_next_segment,
    segment_end,
)
from .func import Func
from .ppoly import PPoly

__all__ = [
    # Functions
    "PPoly",
    "Func",
    # Algorithm
    "bottleneck_analysis",
    "ppoly_min",
    "ppoly_min2",
    "ppoly_min_next_segment",
    "get_speed",
    "segment_end",
    "next_segment_only",
    "last_segment",
]
