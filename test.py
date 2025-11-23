"""Parity checks between legacy ``bottlemod`` and refactored ``bottlemod_new``.

Run this module directly to execute the scenarios that mirror the
``paper_figures_general`` tests from the original project. Each scenario
instantiates both implementations with identical inputs and asserts that the
resulting progress curves and bottleneck traces match.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from bottlemod.func import Func as LegacyFunc
from bottlemod.ppoly import PPoly as LegacyPPoly
from bottlemod.task import Task as LegacyTask
from bottlemod.task import TaskExecution as LegacyTaskExecution
from bottlemod_new.core import Func as NewFunc
from bottlemod_new.core import PPoly as NewPPoly
from bottlemod_new.core import bottleneck_analysis
from bottlemod_new.model import (
    ExecutionEnvironment as NewExecutionEnvironment,
)
from bottlemod_new.model import (
    ResourceType,
)
from bottlemod_new.model import (
    TaskExecution as NewTaskExecution,
)
from bottlemod_new.model import (
    TaskRequirements as NewTaskRequirements,
)
from bottlemod_new.visualization.plot import plot_combined_analysis

# Each scenario entry contains (breakpoints, coefficients) suitable for the
# respective PPoly/Func constructor.
ScenarioDef = Tuple[Sequence[float], Sequence[Sequence[float]]]


def integrate_rate(
    rate: NewPPoly,
    *,
    override_end: float | None = None,
    extra_breaks: Iterable[float] | None = None,
) -> NewPPoly:
    """Convert a piecewise-constant rate function into cumulative form."""

    x_breaks: List[float] = [float(val) for val in rate.x]
    if override_end is not None:
        x_breaks[-1] = override_end
    if extra_breaks:
        start = x_breaks[0]
        end = x_breaks[-1]
        augmented = set(x_breaks)
        for point in extra_breaks:
            if start < point < end:
                augmented.add(float(point))
        x_breaks = sorted(augmented)

    slopes: List[float] = []
    intercepts: List[float] = []
    cumulative_base = 0.0
    for idx in range(len(x_breaks) - 1):
        start = x_breaks[idx]
        end = x_breaks[idx + 1]
        midpoint = (start + end) / 2.0
        rate_value = float(rate(midpoint))
        slopes.append(rate_value)
        intercepts.append(cumulative_base - rate_value * start)
        cumulative_base += rate_value * (end - start)

    return NewPPoly(x_breaks, [slopes, intercepts])


def clone_ppoly(cls, spec: ScenarioDef):
    x, c = spec
    coeffs = [list(segment) for segment in c]
    return cls(list(x), coeffs)


def build_new_execution(
    resource_requirements: Sequence[NewPPoly],
    resource_inputs: Sequence[NewPPoly],
    data_dependencies: Sequence[NewFunc],
    data_inputs: Sequence[NewFunc],
    max_progress: float,
) -> NewTaskExecution:
    requirements = NewTaskRequirements()
    for idx, requirement in enumerate(resource_requirements):
        requirements.add_resource(
            ResourceType.CPU,
            requirement,
            name=f"CPU_{idx}",
        )

    for idx, dependency in enumerate(data_dependencies):
        requirements.add_data(dependency, name=f"Data_{idx}")

    environment = NewExecutionEnvironment()
    for idx, resource_input in enumerate(resource_inputs):
        environment.add_resource(
            ResourceType.CPU,
            resource_input,
            name=f"CPU_{idx}",
        )

    for idx, data_input in enumerate(data_inputs):
        environment.add_data(data_input, name=f"Data_{idx}")

    # Ensure max_progress is propagated when no resources are present.
    if requirements.max_progress == 1.0 and max_progress != 0.0:
        requirements.max_progress = max_progress

    return NewTaskExecution(requirements, environment)


def plot_combined_new(name: str, execution: NewTaskExecution) -> None:
    fig = plot_combined_analysis(
        execution,
        title=f"BottleMod New - {name}",
    )
    filename = name.lower().replace(" ", "_") + "_new_combined.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined_legacy(
    name: str,
    progress_func: LegacyPPoly,
    resource_requirements: Sequence[NewPPoly],
    resource_inputs: Sequence[NewPPoly],
    data_dependencies: Sequence[NewFunc],
    data_inputs: Sequence[NewFunc],
    max_progress: float,
) -> None:
    total_time = float(progress_func.x[-1]) if len(progress_func.x) > 0 else 0.0
    if total_time <= 0:
        total_time = 1.0

    time_arr = np.linspace(0.0, total_time, 200)
    progress_vals = np.array([float(progress_func(t)) for t in time_arr])

    progress_cap = max_progress if max_progress > 0 else progress_vals.max()
    if progress_cap <= 0:
        progress_cap = 1.0

    progress_pct = progress_vals / progress_cap * 100.0

    progress_rate_func = progress_func.derivative()
    progress_rates = np.array(
        [
            float(
                np.nan_to_num(
                    progress_rate_func(t),
                    nan=0.0,
                    neginf=0.0,
                    posinf=0.0,
                )
            )
            for t in time_arr
        ]
    )
    progress_rates[progress_rates < 0.0] = 0.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(time_arr, progress_pct, color="tab:blue", linewidth=2.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Progress (%)")
    ax1.set_title("Progress P(t) over Time", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.set_xlim(0.0, total_time)
    ax1.set_ylim(0.0, 105.0)

    for idx, (req_func, inp_func) in enumerate(
        zip(resource_requirements, resource_inputs)
    ):
        color = f"C{idx}"
        label = f"CPU_{idx}"

        req_derivative = req_func.derivative()
        inp_derivative = inp_func.derivative()

        requirement_vals = []
        for p_val, rate in zip(progress_vals, progress_rates):
            clamped_p = min(
                max(p_val, float(req_derivative.x[0])), float(req_derivative.x[-1])
            )
            per_progress = float(
                np.nan_to_num(
                    req_derivative(clamped_p),
                    nan=0.0,
                    neginf=0.0,
                    posinf=0.0,
                )
            )
            requirement_vals.append(max(0.0, per_progress * rate))

        input_vals = []
        for t in time_arr:
            clamped_t = min(
                max(t, float(inp_derivative.x[0])), float(inp_derivative.x[-1])
            )
            value = float(
                np.nan_to_num(
                    inp_derivative(clamped_t),
                    nan=0.0,
                    neginf=0.0,
                    posinf=0.0,
                )
            )
            input_vals.append(max(0.0, value))

        ax2.plot(
            time_arr,
            requirement_vals,
            color=color,
            linewidth=2,
            linestyle="-",
            alpha=0.9,
            label=f"{label} (required rate)",
        )
        ax2.plot(
            time_arr,
            input_vals,
            color=color,
            linewidth=2,
            linestyle="--",
            alpha=0.7,
            label=f"{label} (input rate)",
        )

    for idx, (dep_func, inp_func) in enumerate(zip(data_dependencies, data_inputs)):
        color = "tab:brown"
        label = f"Data_{idx}"

        dep_derivative = dep_func.derivative()
        inp_derivative = inp_func.derivative()

        requirement_vals = []
        for p_val, rate in zip(progress_vals, progress_rates):
            clamped_p = min(
                max(p_val, float(dep_derivative.x[0])), float(dep_derivative.x[-1])
            )
            per_progress = float(
                np.nan_to_num(
                    dep_derivative(clamped_p),
                    nan=0.0,
                    neginf=0.0,
                    posinf=0.0,
                )
            )
            requirement_vals.append(max(0.0, per_progress * rate))

        input_vals = []
        for t in time_arr:
            clamped_t = min(
                max(t, float(inp_derivative.x[0])), float(inp_derivative.x[-1])
            )
            value = float(
                np.nan_to_num(
                    inp_derivative(clamped_t),
                    nan=0.0,
                    neginf=0.0,
                    posinf=0.0,
                )
            )
            input_vals.append(max(0.0, value))

        ax2.plot(
            time_arr,
            requirement_vals,
            color=color,
            linewidth=2,
            linestyle="-",
            alpha=0.9,
            label=f"{label} (required rate)",
        )
        ax2.plot(
            time_arr,
            input_vals,
            color=color,
            linewidth=2,
            linestyle="--",
            alpha=0.7,
            label=f"{label} (input rate)",
        )

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Resource Rate")
    ax2.set_title("Input and Requirements Analysis", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.set_xlim(0.0, total_time)
    ax2.legend(loc="best", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    filename = name.lower().replace(" ", "_") + "_legacy_combined.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compare_scenario(
    name: str,
    out_cpu_defs: Sequence[ScenarioDef],
    in_cpu_defs: Sequence[ScenarioDef],
    out_data_defs: Sequence[ScenarioDef],
    in_data_defs: Sequence[ScenarioDef],
) -> None:
    print(f"Scenario: {name}")

    legacy_out_cpu = [clone_ppoly(LegacyPPoly, spec) for spec in out_cpu_defs]
    legacy_out_data = [clone_ppoly(LegacyFunc, spec) for spec in out_data_defs]
    legacy_in_cpu = [clone_ppoly(LegacyPPoly, spec) for spec in in_cpu_defs]
    legacy_in_data = [clone_ppoly(LegacyFunc, spec) for spec in in_data_defs]

    max_progress = float(legacy_out_data[0](legacy_out_data[0].x[-1]))
    for poly in legacy_out_cpu:
        poly.x[-1] = max_progress

    legacy_progress, legacy_bottlenecks = LegacyTaskExecution(
        LegacyTask(legacy_out_cpu, legacy_out_data),
        legacy_in_cpu,
        legacy_in_data,
    ).get_result()

    new_out_cpu = [clone_ppoly(NewPPoly, spec) for spec in out_cpu_defs]
    new_in_cpu = [clone_ppoly(NewPPoly, spec) for spec in in_cpu_defs]
    new_out_data = [clone_ppoly(NewFunc, spec) for spec in out_data_defs]
    new_in_data = [clone_ppoly(NewFunc, spec) for spec in in_data_defs]

    resource_requirements = [
        integrate_rate(rate, override_end=max_progress) for rate in new_out_cpu
    ]
    global_breaks = sorted({float(val) for rate in new_in_cpu for val in rate.x})
    resource_inputs = [
        integrate_rate(rate, extra_breaks=global_breaks) for rate in new_in_cpu
    ]

    new_progress, new_bottlenecks = bottleneck_analysis(
        resource_requirements,
        resource_inputs,
        new_out_data,
        new_in_data,
    )

    assert legacy_progress.__str__() == new_progress.__str__(), (
        "Progress mismatch",
        legacy_progress,
        new_progress,
    )
    assert tuple(legacy_bottlenecks) == tuple(new_bottlenecks), (
        "Bottleneck mismatch",
        legacy_bottlenecks,
        new_bottlenecks,
    )

    print("  progress match ✔")
    print("  bottlenecks match ✔")

    plot_progress(name, legacy_progress, new_progress)

    new_execution = build_new_execution(
        resource_requirements,
        resource_inputs,
        new_out_data,
        new_in_data,
        max_progress,
    )
    plot_combined_new(name, new_execution)

    plot_combined_legacy(
        name,
        legacy_progress,
        resource_requirements,
        resource_inputs,
        new_out_data,
        new_in_data,
        max_progress,
    )


def plot_progress(
    name: str, legacy_progress: LegacyPPoly, new_progress: NewPPoly
) -> None:
    """Plot the progress curve for both implementations and save to disk."""

    time_domain = np.linspace(new_progress.x[0], new_progress.x[-1], 1000)
    legacy_values = legacy_progress(time_domain)
    new_values = new_progress(time_domain)

    plt.figure()
    plt.plot(time_domain, legacy_values, label="legacy", linewidth=2)
    plt.plot(time_domain, new_values, label="bottlemod_new", linestyle="--")
    plt.xlabel("time")
    plt.ylabel("progress")
    plt.title(f"Progress Comparison - {name}")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    filename = name.lower().replace(" ", "_") + "_progress.png"
    plt.savefig(filename)
    plt.close()


def main() -> None:
    scenario1_out_cpu = [([0, 10000], [[1]])] * 3
    scenario1_in_cpu = [
        ([0, 100], [[3]]),
        ([0, 60, 100], [[10, 1.5]]),
        ([0, 40, 100], [[0.8, 10]]),
    ]
    scenario1_out_data = [([0, 100], [[1, 0]])] * 3
    scenario1_in_data = [
        ([0, 100], [[1, 0]]),
        ([0, 50, 100], [[20], [100]]),
        ([0, 63.245553203367586639977870888654], [[0.025, 0, 0]]),
    ]

    scenario2_out_cpu = [
        ([0, 20, 60, 10000], [[0.8, 4.0 / 3.0, 0.8 / 3.0]]),
        ([0, 40, 90, 10000], [[0.4, 1.0, 0.3]]),
        ([0, 30, 80, 10000], [[2.0, 0.8, 0.1]]),
    ]
    scenario2_in_cpu = [
        ([0, 100], [[4]]),
        ([0, 60, 100], [[5, 1.5]]),
        ([0, 40, 100], [[1.6, 6.5]]),
    ]
    scenario2_out_data = [([0, 100], [[1, 0]])] * 3
    scenario2_in_data = [
        ([0, 100], [[1, 0]]),
        ([0, 50, 100], [[20], [100]]),
        (
            [0, 63.245553203367586639977870888654, 100],
            [[0.025, 0, 0], [0, 0, 100]],
        ),
    ]

    compare_scenario(
        "Test1",
        scenario1_out_cpu,
        scenario1_in_cpu,
        scenario1_out_data,
        scenario1_in_data,
    )
    compare_scenario(
        "Test1_Plot",
        scenario2_out_cpu,
        scenario2_in_cpu,
        scenario2_out_data,
        scenario2_in_data,
    )

    example_out_cpu = [([0, 5000, 10000], [[1, 2]])]
    example_in_cpu = [([0, 10000], [[2]])]
    example_out_data = [([0, 10000], [[1, 0]])]
    example_in_data = [([0, 10000], [[1, 0]])]

    compare_scenario(
        "ExampleSimple",
        example_out_cpu,
        example_in_cpu,
        example_out_data,
        example_in_data,
    )


if __name__ == "__main__":
    main()
