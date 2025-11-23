#!/usr/bin/env python3
"""
Simple Example - BottleMod New API

Demonstrates the basic usage of the refactored BottleMod framework.
"""

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


def example1_simple_cpu(visualize=False):
    """Example 1: Simple CPU-only task."""

    # Requirements
    cpu_requirement_1 = PPoly(
        [0, 5000, 10000],
        [[1, 2], [2, 2]],
    )
    task = TaskRequirements().add_resource(
        resource_type=ResourceType.CPU,
        requirement_func=cpu_requirement_1,
        name="CPU_0_requirement_1",
    )

    # Resource inputs
    cpu_input_1 = PPoly([0, 10000], [[2], [2]])
    env = ExecutionEnvironment().add_resource(
        resource_type=ResourceType.CPU,
        input_func=cpu_input_1,
        name="CPU_0_input_1",
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
        fig.savefig("example_simple_visualization.png", dpi=150, bbox_inches="tight")
        print("Saved: example_simple_visualization.png")

    return execution


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  BottleMod - Simple Examples")
    print("#" * 70)

    example1_simple_cpu(visualize=True)
