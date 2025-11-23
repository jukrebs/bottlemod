"""
BottleMod Analysis Module

This module provides tools for analyzing and visualizing bottleneck results:
- Bottleneck timeline analysis
- Time distribution calculations
- Comparison utilities
- Visualization (future)
"""

from .bottleneck import (
    compute_bottleneck_time,
    compute_bottleneck_timeline_segments,
    print_bottleneck_timeline,
    print_bottleneck_summary,
    analyze_critical_path,
    compare_executions,
)

__all__ = [
    "compute_bottleneck_time",
    "compute_bottleneck_timeline_segments",
    "print_bottleneck_timeline",
    "print_bottleneck_summary",
    "analyze_critical_path",
    "compare_executions",
]
