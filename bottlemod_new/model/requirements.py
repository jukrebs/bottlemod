"""
Task Requirements Module

Defines what a task requires to execute, separating:
- Resource requirements (rate-limiting: CPU, cache, disk, etc.)
- Data dependencies (availability prerequisites)
"""

from typing import List, Optional

from ..core import (
    Func,
    PPoly,
)
from .resources import DataDependency, ResourceRequirement, ResourceType


class TaskRequirements:
    """
    Defines what a task requires to execute.

    This class separates two types of constraints:
    1. Resource requirements: Such as (CPU, cache, disk, etc.)
    2. Data dependencies: Availability prerequisites
       - Task cannot proceed without required data being present

    Attributes
    ----------
    resource_requirements : List[ResourceRequirement]
        List of resource requirements with their cumulative functions R(p)
    data_dependencies : List[DataDependency]
        List of data dependencies with their cumulative functions D(p)
    max_progress : float
        Maximum progress value (default: 1.0)

    """

    def __init__(
        self,
        resource_requirements: Optional[List[ResourceRequirement]] = None,
        data_dependencies: Optional[List[DataDependency]] = None,
        max_progress: float = 1.0,
    ):
        """
        Initialize task requirements.

        Parameters
        ----------
        resource_requirements : List[ResourceRequirement], optional
            List of resource requirements (default: empty list)
        data_dependencies : List[DataDependency], optional
            List of data dependencies (default: empty list)
        max_progress : float, optional
            Maximum progress value (default: 1.0)
        """
        self.resource_requirements = resource_requirements or []
        self.data_dependencies = data_dependencies or []
        self.max_progress = max_progress

        # Validate requirements
        self._validate()

    def _validate(self):
        """Ensure all requirements have compatible progress ranges."""
        if not self.resource_requirements:
            return

        # Check that all resource requirements end at same progress
        first_max = self.resource_requirements[0].requirement_func.x[-1]
        for req in self.resource_requirements:
            req_max = req.requirement_func.x[-1]
            if abs(req_max - first_max) > 1e-10:
                raise ValueError(
                    f"All resource requirements must have same max progress. "
                    f"Got {req}: max={req_max}, expected {first_max}"
                )

        # Check data dependencies if present
        for dep in self.data_dependencies:
            dep_max = dep.dependency_func(dep.dependency_func.x[-1])
            if abs(dep_max - first_max) > 1e-10:
                raise ValueError(
                    f"Data dependency {dep} must reach same max progress as resources. "
                    f"Got {dep_max}, expected {first_max}"
                )

        # Update max_progress if not explicitly set
        if self.max_progress == 1.0 and first_max != 1.0:
            self.max_progress = first_max

    def add_resource(
        self,
        resource_type: ResourceType,
        requirement_func: PPoly,
        name: str = "",
    ) -> "TaskRequirements":
        """
        Add a resource requirement (builder pattern).

        Parameters
        ----------
        resource_type : ResourceType
            Type of resource (CPU, CACHE, DISK, etc.)
        requirement_func : PPoly
            Requirement function R(p)
        name : str, optional
            Human-readable name for this requirement

        Returns
        -------
        TaskRequirements
            Self, for method chaining

        Examples
        --------
        >>> requirements = TaskRequirements() \\
        ...     .add_resource(ResourceType.CPU, cpu_func, "Worker_CPU") \\
        ...     .add_resource(ResourceType.DISK, disk_func, "SSD")
        """
        req = ResourceRequirement(resource_type, requirement_func, name)
        self.resource_requirements.append(req)
        self._validate()  # Update max_progress after adding resource
        return self

    def add_data(
        self,
        dependency_func: Func,
        name: str = "",
    ) -> "TaskRequirements":
        """
        Add a data dependency (builder pattern).

        Parameters
        ----------
        dependency_func : Func
            Monotonic inverse data requirement R^{-1}_{D,k}(n) mapping
            available data ``n`` to achievable progress ``p``
        name : str, optional
            Human-readable name for this dependency

        Returns
        -------
        TaskRequirements
            Self, for method chaining

        Examples
        --------
        >>> requirements = TaskRequirements() \\
        ...     .add_resource(ResourceType.CPU, cpu_func) \\
        ...     .add_data(dataset_func, "Training_Data")
        """
        dep = DataDependency(dependency_func, name)
        self.data_dependencies.append(dep)
        self._validate()  # Update max_progress after adding data
        return self

    def get_resource(self, name: str) -> Optional[ResourceRequirement]:
        """
        Get resource requirement by name.

        Parameters
        ----------
        name : str
            Name of the resource to find

        Returns
        -------
        ResourceRequirement or None
            The matching requirement, or None if not found
        """
        for req in self.resource_requirements:
            if req.name == name:
                return req
        return None

    def get_resources_by_type(
        self, resource_type: ResourceType
    ) -> List[ResourceRequirement]:
        """
        Get all resource requirements of a specific type.

        Parameters
        ----------
        resource_type : ResourceType
            Type of resource to find

        Returns
        -------
        List[ResourceRequirement]
            All matching requirements
        """
        return [
            req
            for req in self.resource_requirements
            if req.resource_type == resource_type
        ]

    def __repr__(self):
        res_count = len(self.resource_requirements)
        data_count = len(self.data_dependencies)
        return f"TaskRequirements(resources={res_count}, data={data_count}, max_progress={self.max_progress})"

    def __str__(self):
        lines = ["Task Requirements:"]

        if self.resource_requirements:
            lines.append("  Resources:")
            for req in self.resource_requirements:
                lines.append(f"    - {req}")

        if self.data_dependencies:
            lines.append("  Data Dependencies:")
            for dep in self.data_dependencies:
                lines.append(f"    - {dep}")

        lines.append(f"  Max Progress: {self.max_progress}")

        return "\n".join(lines)
