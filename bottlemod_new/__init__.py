"""BottleMod Cache-Aware"""

# Core components
# Builders
from . import builders

# Analysis tools
from .analysis import (
    analyze_critical_path,
    compare_executions,
    compute_bottleneck_time,
    compute_bottleneck_timeline_segments,
    print_bottleneck_summary,
    print_bottleneck_timeline,
)

# Core algorithms and function types
from .core import (
    Func,
    # Functions
    PPoly,
    # Algorithm (advanced users)
    bottleneck_analysis,
    ppoly_min,
)

# Model components and resource definitions
from .model import (
    DataDependency,
    DataInput,
    ExecutionEnvironment,
    Input,
    Resource,
    ResourceInput,
    ResourceRequirement,
    ResourceType,
    TaskExecution,
    TaskRequirements,
)

# Visualization
from .visualization import (
    plot_bottleneck_analysis,
    plot_combined_analysis,
    plot_progress_timeline,
)

__version__ = "0.0.1"

__all__ = [
    # Core
    "PPoly",
    "Func",
    "ResourceType",
    "ResourceRequirement",
    "ResourceInput",
    "DataDependency",
    "DataInput",
    "Resource",
    "Input",
    # Model
    "TaskRequirements",
    "ExecutionEnvironment",
    "TaskExecution",
    # Analysis
    "compute_bottleneck_time",
    "compute_bottleneck_timeline_segments",
    "print_bottleneck_timeline",
    "print_bottleneck_summary",
    "analyze_critical_path",
    "compare_executions",
    # Visualization
    "plot_bottleneck_analysis",
    "plot_progress_timeline",
    "plot_combined_analysis",
    # Builders module
    "builders",
    # Algorithm (advanced)
    "bottleneck_analysis",
    "ppoly_min",
]
