"""
Task Execution Module

Executes Bottlemod run/analysis for a task in an environment.
"""

import bisect
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..analysis.bottleneck import compute_bottleneck_time
from ..core import (
    PPoly,
    bottleneck_analysis,
)
from .environment import ExecutionEnvironment
from .requirements import TaskRequirements
from .resources import ResourceInput, ResourceRequirement, ResourceType


class TaskExecution:
    """
    Represents execution of a task in a given environment.

    This class brings together task requirements and execution environment
    to perform bottleneck analysis, producing:
    1. Progress function P(t) - how progress evolves over time
    2. Bottleneck timeline - which resource limits progress when

    Attributes
    ----------
    requirements : TaskRequirements
        What the task needs
    environment : ExecutionEnvironment
        What the hardware provides
    progress_function : PPoly
        Progress P(t) over time (computed)
    bottlenecks : List[]
        Which constraint is active in each time segment (computed)
    """

    def __init__(
        self,
        requirements: TaskRequirements,
        environment: ExecutionEnvironment,
    ):
        """
        Initialize task execution and perform bottleneck analysis.

        Parameters
        ----------
        requirements : TaskRequirements
            Task requirements (what it needs)
        environment : ExecutionEnvironment
            Execution environment (what's available)

        Raises
        ------
        ValueError
            If requirements cannot be matched to environment
        """
        self.requirements = requirements
        self.environment = environment

        # Match requirements to environment
        self.resource_matches, self.data_matches = (
            ExecutionEnvironment.match_requirements(requirements, environment)
        )

        resource_matches_by_type: Dict[
            ResourceType, List[Tuple[ResourceRequirement, ResourceInput]]
        ] = defaultdict(list)
        for req, inp in self.resource_matches:
            resource_matches_by_type[req.resource_type].append((req, inp))

        self.resource_matches_by_type: Dict[
            ResourceType, List[Tuple[ResourceRequirement, ResourceInput]]
        ] = dict(resource_matches_by_type)

        # TODO: Other resource types and data dependencies
        self.cpu_requirements = self.requirements.get_resources_by_type(
            ResourceType.CPU
        )

        # TODO: Other input types and data inputs/availability
        self.cpu_environment = self.environment.get_inputs_by_type(ResourceType.CPU)

        # Results
        self.progress_function: Optional[PPoly] = None
        self.bottlenecks: List = []

        # Run bottleneck analysis
        self._calculate()

    def _calculate(self):
        """
        Perform bottleneck analysis using core algorithm.

        This extracts requirement/input functions from matched pairs,
        calls the core bottleneck_analysis algorithm, and converts
        the returned indices back to resource/data objects.
        """
        # Extract requirement functions
        resource_reqs = []
        resource_inputs = []
        for req, inp in self.resource_matches:
            resource_reqs.append(req.requirement_func)
            resource_inputs.append(inp.input_func)

        # Extract data functions
        # Data dependencies map "available data" -> "max achievable progress"
        # (see paper §3.1). Leave empty when the task has no data prerequisites.
        data_deps = [dep.dependency_func for dep, _ in self.data_matches]
        data_inputs = [inp.input_func for _, inp in self.data_matches]

        # Run core algorithm
        progress_func, bottleneck_indices = bottleneck_analysis(
            resource_requirements=resource_reqs,
            resource_inputs=resource_inputs,
            data_dependencies=data_deps,
            data_inputs=data_inputs,
        )

        # Store progress function
        self.progress_function = progress_func

        # Convert indices to objects
        self.bottlenecks = [
            self._interpret_bottleneck_index(idx) for idx in bottleneck_indices
        ]

    def _interpret_bottleneck_index(self, idx: int):
        """
        Convert bottleneck index to ResourceRequirement or DataDependency object.

        Parameters
        ----------
        idx : int
            Bottleneck index from algorithm
            - Negative: resource bottleneck (-1-k means resource k)
            - Non-negative: data dependency bottleneck (index into data_matches)

        Returns
        -------
        ResourceRequirement or DataDependency
            The bottleneck object
        """
        if idx < 0:
            # Resource bottleneck
            resource_idx = -idx - 1
            if 0 <= resource_idx < len(self.resource_matches):
                return self.resource_matches[resource_idx][
                    0
                ]  # Return requirement object
            else:
                return None  # Shouldn't happen
        else:
            # Data dependency bottleneck
            if 0 <= idx < len(self.data_matches):
                return self.data_matches[idx][0]  # Return dependency object
            else:
                return None  # Shouldn't happen

    def total_time(self) -> float:
        """
        Get total execution time to reach max progress.

        Returns
        -------
        float
            Time in seconds to complete task
        """
        if self.progress_function is None:
            return 0.0
        result = self.progress_function.solve(self.requirements.max_progress)
        # Filter for positive/forward times only
        positive_results = result[result >= 0]
        return float(positive_results[0]) if len(positive_results) > 0 else 0.0

    def bottleneck_summary(self) -> Dict[ResourceType, float]:
        """
        Calculate time spent in each resource type bottleneck.

        Returns
        -------
        Dict[ResourceType, float]
            Time spent bottlenecked on each resource type
        """

        return compute_bottleneck_time(self)

    def get_progress_at_time(self, t: float) -> float:
        """
        Get progress achieved at a specific time.

        Parameters
        ----------
        t : float
            Time to query

        Returns
        -------
        float
            Progress at time t
        """
        if self.progress_function is None:
            return 0.0
        return self.progress_function(t)

    def get_bottleneck_at_time(self, t: float):
        """
        Get which resource/dependency is bottleneck at time t.

        Parameters
        ----------
        t : float
            Time to query

        Returns
        -------
        ResourceRequirement or DataDependency
            The bottleneck at time t
        """

        if not self.bottlenecks or self.progress_function is None:
            return None

        # Find which segment t falls in
        idx = bisect.bisect_right(self.progress_function.x, t) - 1
        if 0 <= idx < len(self.bottlenecks):
            return self.bottlenecks[idx]
        return None

    def __repr__(self):
        if self.progress_function is None:
            return "TaskExecution(not computed)"
        total_time = float(self.total_time())
        return f"TaskExecution(total_time={total_time:.2f}s, segments={len(self.bottlenecks)})"

    def __str__(self):
        if self.progress_function is None:
            return "TaskExecution: Not computed"

        lines = ["Task Execution Results:"]
        lines.append(f"  Total time: {float(self.total_time()):.2f}s")
        lines.append(f"  Max progress: {self.requirements.max_progress}")
        lines.append(f"  Bottleneck segments: {len(self.bottlenecks)}")

        return "\n".join(lines)
