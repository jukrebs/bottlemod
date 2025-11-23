"""
Execution Environment Module

Defines available resources in the execution environment (what the hardware can provide).
"""

from typing import List, Optional, Tuple

from ..core import Func, PPoly
from .requirements import TaskRequirements
from .resources import (
    DataDependency,
    DataInput,
    ResourceInput,
    ResourceRequirement,
    ResourceType,
)


class ExecutionEnvironment:
    """
    Defines available resources in the execution environment.

    This represents what the hardware/system can provide:
    - Resource inputs: Available bandwidth/capacity (CPU, cache, disk, network)
    - Data inputs: Data availability over time

    Attributes
    ----------
    resource_inputs : List[ResourceInput]
        List of available resources with their cumulative functions I(t)
    data_inputs : List[DataInput]
        List of data inputs with their availability functions I_D(t)

    """

    def __init__(
        self,
        resource_inputs: Optional[List[ResourceInput]] = None,
        data_inputs: Optional[List[DataInput]] = None,
    ):
        """
        Initialize execution environment.

        Parameters
        ----------
        resource_inputs : List[ResourceInput], optional
            List of available resources (default: empty list)
        data_inputs : List[DataInput], optional
            List of data inputs (default: empty list)
        """
        self.resource_inputs = resource_inputs or []
        self.data_inputs = data_inputs or []

    def add_resource(
        self,
        resource_type: ResourceType,
        input_func: PPoly,
        name: str = "",
    ) -> "ExecutionEnvironment":
        """
        Add a resource input (builder pattern).

        Parameters
        ----------
        resource_type : ResourceType
            Type of resource being provided
        input_func : PPoly
            Cumulative input function I(t)
        name : str, optional
            Human-readable name for this resource

        Returns
        -------
        ExecutionEnvironment
            Self, for method chaining
        """
        inp = ResourceInput(resource_type, input_func, name)
        self.resource_inputs.append(inp)
        return self

    def add_data(
        self,
        input_func: Func,
        name: str = "",
    ) -> "ExecutionEnvironment":
        """
        Add a data input (builder pattern).

        Parameters
        ----------
        input_func : Func
            Cumulative data availability function I_D(t)
        name : str, optional
            Human-readable name for this data source

        Returns
        -------
        ExecutionEnvironment
            Self, for method chaining
        """
        inp = DataInput(input_func, name)
        self.data_inputs.append(inp)
        return self

    def get_input(self, name: str) -> Optional[ResourceInput]:
        """
        Get resource input by name.

        Parameters
        ----------
        name : str
            Name of the resource to find

        Returns
        -------
        ResourceInput or None
            The matching input, or None if not found
        """
        for inp in self.resource_inputs:
            if inp.name == name:
                return inp
        return None

    def get_inputs_by_type(self, resource_type: ResourceType) -> List[ResourceInput]:
        """
        Get all inputs matching a resource type.

        Parameters
        ----------
        resource_type : ResourceType
            Type of resource to find

        Returns
        -------
        List[ResourceInput]
            All matching inputs
        """
        return [
            inp for inp in self.resource_inputs if inp.resource_type == resource_type
        ]

    @staticmethod
    def match_requirements(
        requirements: TaskRequirements,
        environment: "ExecutionEnvironment",
    ) -> Tuple[
        List[Tuple[ResourceRequirement, ResourceInput]],
        List[Tuple[DataDependency, DataInput]],
    ]:
        """
        match requirements to environment inputs based on resource type.

        This is the key matching algorithm that pairs task requirements
        with environment capabilities. Matching is performed by
        ``ResourceType`` (optionally refined by human-readable names).
        We deliberately do *not* compare the domains of requirement
        functions (progress) with input functions (time), because those
        quantities live in different spaces per the BottleMod model.

            Parameters
            ----------
            requirements : TaskRequirements
                Task requirements to match
            environment : ExecutionEnvironment
                Environment providing resources

            Returns
            -------
            Tuple[List[Tuple[ResourceRequirement, ResourceInput]], List[Tuple[DataDependency, DataInput]]]
                Two lists:
                - Resource matches: (requirement, input) pairs
                - Data matches: (dependency, input) pairs

            Raises
            ------
            ValueError
                If a requirement cannot be matched to an input with the same
                type

        """
        resource_matches: List[Tuple[ResourceRequirement, ResourceInput]] = []
        data_matches: List[Tuple[DataDependency, DataInput]] = []

        # Organise inputs by resource type to speed up lookups
        inputs_by_type: dict[ResourceType, List[ResourceInput]] = {}
        for inp in environment.resource_inputs:
            inputs_by_type.setdefault(inp.resource_type, []).append(inp)

        # Match resources by type (optionally by name)
        for req in requirements.resource_requirements:
            candidates = inputs_by_type.get(req.resource_type, [])
            if not candidates:
                raise ValueError(
                    f"No environment input found for requirement '{req}' "
                    f"with type {req.resource_type.name}"
                )

            # Prefer inputs with matching names when provided
            ordered_candidates: List[Tuple[int, ResourceInput]] = []
            if req.name:
                ordered_candidates.extend(
                    (
                        idx,
                        inp,
                    )
                    for idx, inp in enumerate(candidates)
                    if inp.name == req.name
                )

            if not ordered_candidates:
                ordered_candidates = list(enumerate(candidates))

            matched_index: Optional[int] = None
            for idx, _ in ordered_candidates:
                matched_index = idx
                break

            if matched_index is None:
                raise ValueError(f"No environment input found for requirement '{req}'")

            inp = candidates[matched_index]
            resource_matches.append((req, inp))

            # Prevent the same input from being used multiple times by default
            del candidates[matched_index]
            if not candidates:
                del inputs_by_type[req.resource_type]

        # Match data dependencies by name
        for dep in requirements.data_dependencies:
            if not dep.name:
                raise ValueError(f"Data dependency {dep} must have a name for matching")

            # Find matching data input
            data_inp = None
            for inp in environment.data_inputs:
                if inp.name == dep.name:
                    data_inp = inp
                    break

            if data_inp is None:
                raise ValueError(
                    f"No environment data input found for dependency '{dep.name}'"
                )

            data_matches.append((dep, data_inp))

        return resource_matches, data_matches

    def __repr__(self):
        res_count = len(self.resource_inputs)
        data_count = len(self.data_inputs)
        return f"ExecutionEnvironment(resources={res_count}, data={data_count})"

    def __str__(self):
        lines = ["Execution Environment:"]

        if self.resource_inputs:
            lines.append("  Resource Inputs:")
            for inp in self.resource_inputs:
                lines.append(f"    - {inp}")

        if self.data_inputs:
            lines.append("  Data Inputs:")
            for inp in self.data_inputs:
                lines.append(f"    - {inp}")

        return "\n".join(lines)
