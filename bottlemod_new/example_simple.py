#!/usr/bin/env python3
"""
Simple Example - BottleMod New API

Demonstrates the basic usage of the refactored BottleMod framework.
"""

from pathlib import Path

# Use relative imports
from bottlemod_new import (
    TaskExecution,
    plot_combined_analysis,
    print_bottleneck_summary,
    print_bottleneck_timeline,
)
from bottlemod_new.core import PPoly
from bottlemod_new.model.environment import ExecutionEnvironment
from bottlemod_new.model.requirements import (
    TaskRequirements,
)
from bottlemod_new.model.resources import ResourceType

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def example1_simple_cpu(visualize=False):
    """Example 1: Simple CPU-only task."""

    # Requirements: progress 0 to 100
    # First half (0-50): R'(p) = 1 (needs 1 CPU per progress)
    # Second half (50-100): R'(p) = 2 (needs 2 CPU per progress)
    cpu_requirement_1 = PPoly([0, 50, 100], [[1, 2], [2, -48]])

    # Data dependency: 1 data unit = 1 progress unit
    # Need 100 data total to reach progress 100
    data_dep = PPoly([0, 100], [[1], [0]])

    task = (
        TaskRequirements()
        .add_resource(
            resource_type=ResourceType.CPU,
            requirement_func=cpu_requirement_1,
            name="CPU_0",
        )
        .add_data(
            dependency_func=data_dep,
            name="Data_0",
        )
    )

    # Resource inputs
    # CPU provides rate 2 per time unit
    cpu_input_1 = PPoly([0, 1000], [[2], [2]])

    # Data: 100 units instantly available (constant), does not limit progress
    # CPU speed is max 2, so CPU is the bottleneck, not data
    data_input_1 = PPoly([0, 1000], [[0], [100]])

    env = (
        ExecutionEnvironment()
        .add_resource(
            resource_type=ResourceType.CPU,
            input_func=cpu_input_1,
            name="CPU_0",
        )
        .add_data(
            input_func=data_input_1,
            name="Data_0",
        )
    )

    # Execute bottleneck analysis
    execution = TaskExecution(task, env)

    # Print results
    print_bottleneck_timeline(execution)
    print_bottleneck_summary(execution)

    # Generate visualization if requested
    if visualize:
        print("\nGenerating visualization...")
        fig = plot_combined_analysis(execution, title="Simple CPU Task Analysis")
        out_path = FIGURES_DIR / "example_simple_visualization.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return execution


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  BottleMod - Simple Examples")
    print("#" * 70)

    example1_simple_cpu(visualize=True)
