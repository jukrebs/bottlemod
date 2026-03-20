"""
BottleMod-CA: Cache-Aware Extension for BottleMod

This module extends BottleMod to model storage hierarchy effects (caches, tiered storage)
while preserving the original separation of concerns:
- Process descriptors (requirements): time-independent relations over progress p
- Environment descriptors (inputs): time-dependent availability/allocations over real time t

Key abstractions:
- LogicalAccessProfile: Process-side logical access requirements (A^r, A^w)
- StorageTier: Environment-side tier with bandwidth limits
- TierMapping: Environment-side cache/tier hit-rate functions (H^r, H^w)
- CacheBehaviorModel: Computes H from reuse descriptors and cache capacities

After the mapping step, we derive standard BottleMod resource requirement functions R_{R,l}(p)
per tier, enabling reuse of the existing progress calculation algorithm.

References:
- BottleMod (ICPE'25): Modeling Data Flows and Tasks for Fast Bottleneck Analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, cast, override

import numpy

from bottlemod.func import Func
from bottlemod.ppoly import PPoly

if TYPE_CHECKING:
    from bottlemod.task import Task


# =============================================================================
# Enumerations
# =============================================================================


class AccessType(Enum):
    """Type of storage access."""

    READ = auto()
    WRITE = auto()


class ResourceType(Enum):
    """Type of storage resource constraint."""

    BANDWIDTH = auto()  # bytes/s


# =============================================================================
# Process-Side: Logical Access Profile
# =============================================================================


@dataclass
class LogicalAccessProfile:
    """
    Process-side logical access requirements for a dataset k, independent of time.

    All functions are cumulative over progress p (monotone non-decreasing).

    Attributes:
        name: Identifier for this dataset/input
        A_read: A^r_k(p) - cumulative logical bytes read up to progress p
        A_write: A^w_k(p) - cumulative logical bytes written up to progress p

    Example - Sequential streaming reads:
        A_read = PPoly([0, 100], [[1000]])  # 1000 bytes per progress unit
    """

    name: str
    A_read: PPoly
    A_write: Optional[PPoly] = None

    def __post_init__(self):
        """Validate and set defaults for access profile."""
        # Ensure writes default to zero if not specified
        if self.A_write is None:
            self.A_write = PPoly([self.A_read.x[0], self.A_read.x[-1]], [[0]])

    @classmethod
    def sequential_read(
        cls, name: str, total_bytes: float, max_progress: float
    ) -> LogicalAccessProfile:
        """
        Create a profile for sequential reads (streaming).

        Args:
            name: Dataset identifier
            total_bytes: Total bytes to read
            max_progress: Maximum progress value

        Returns:
            LogicalAccessProfile with linear A^r
        """
        bytes_per_progress = total_bytes / max_progress

        A_read = PPoly([0, max_progress], [[bytes_per_progress], [0]])

        return cls(
            name=name,
            A_read=A_read,
        )

    @classmethod
    def random_read(
        cls,
        name: str,
        total_bytes: float,
        max_progress: float,
    ) -> LogicalAccessProfile:
        """
        Create a profile for random reads (e.g., database lookups).

        Args:
            name: Dataset identifier
            total_bytes: Total bytes to read
            max_progress: Maximum progress value

        Returns:
            LogicalAccessProfile with linear A^r
        """
        bytes_per_progress = total_bytes / max_progress

        A_read = PPoly([0, max_progress], [[bytes_per_progress], [0]])

        return cls(
            name=name,
            A_read=A_read,
        )

    @classmethod
    def piecewise(
        cls, name: str, phases: List[Tuple[float, float, float]]
    ) -> LogicalAccessProfile:
        """
        Create a piecewise profile for multi-phase workloads.

        Args:
            name: Dataset identifier
            phases: List of (start_p, end_p, bytes_rate) tuples
                   Each phase has a constant byte rate per progress unit.

        Returns:
            LogicalAccessProfile with piecewise linear A^r
        """
        if not phases:
            raise ValueError("At least one phase required")

        x_points = [phases[0][0]]
        a_coeffs = []

        for start_p, end_p, bytes_rate in phases:
            x_points.append(end_p)
            a_coeffs.append([bytes_rate])

        A_read = PPoly(x_points, [a_coeffs[i] for i in range(len(a_coeffs))])
        A_read = A_read.antiderivative()
        offset = A_read(x_points[0])
        A_read = A_read - offset

        return cls(name=name, A_read=A_read)

    def get_derivative(self, access_type: AccessType) -> PPoly:
        """Get the derivative (rate) function for the specified access type."""
        if access_type == AccessType.READ:
            return self.A_read.derivative()
        else:
            assert self.A_write is not None
            return self.A_write.derivative()


# =============================================================================
# Environment-Side: Storage Tiers
# =============================================================================


@dataclass
class StorageTier:
    """
    Environment-side storage tier with resource input functions.

    Attributes:
        name: Tier identifier (e.g., "LLC", "DRAM", "SSD", "HDD")
        tier_index: Index in hierarchy (0 = fastest, higher = slower)
        I_bw_read: I^{bw,r}_j(t) - read bandwidth available (bytes/s)
        I_bw_write: I^{bw,w}_j(t) - write bandwidth available (bytes/s)
        capacity: Effective capacity in bytes (for cache hit rate calculation)
    """

    name: str
    tier_index: int
    I_bw_read: PPoly
    I_bw_write: Optional[PPoly] = None
    capacity: Optional[float] = None

    def __post_init__(self):
        """Set defaults for unspecified resource functions."""
        time_range = [self.I_bw_read.x[0], self.I_bw_read.x[-1]]

        if self.I_bw_write is None:
            self.I_bw_write = PPoly(time_range, self.I_bw_read.c.copy())

    @classmethod
    def memory(
        cls,
        name: str = "DRAM",
        bandwidth_GBps: float = 25.0,
        capacity_GB: float = 16.0,
        time_range: Tuple[float, float] = (0, 1e6),
    ) -> StorageTier:
        """Create a memory tier with typical DDR4 characteristics."""
        bw = bandwidth_GBps * 1e9  # Convert to bytes/s
        cap = capacity_GB * 1e9
        return cls(
            name=name,
            tier_index=0,
            I_bw_read=PPoly([time_range[0], time_range[1]], [[bw]]),
            capacity=cap,
        )

    @classmethod
    def nvme_ssd(
        cls,
        name: str = "NVMe",
        bandwidth_GBps: float = 3.0,
        capacity_GB: float = 500.0,
        time_range: Tuple[float, float] = (0, 1e6),
    ) -> StorageTier:
        """Create an NVMe SSD tier with typical characteristics."""
        bw = bandwidth_GBps * 1e9
        cap = capacity_GB * 1e9
        return cls(
            name=name,
            tier_index=1,
            I_bw_read=PPoly([time_range[0], time_range[1]], [[bw]]),
            capacity=cap,
        )

    @classmethod
    def sata_ssd(
        cls,
        name: str = "SATA_SSD",
        bandwidth_MBps: float = 500.0,
        capacity_GB: float = 1000.0,
        time_range: Tuple[float, float] = (0, 1e6),
    ) -> StorageTier:
        """Create a SATA SSD tier with typical characteristics."""
        bw = bandwidth_MBps * 1e6
        cap = capacity_GB * 1e9
        return cls(
            name=name,
            tier_index=2,
            I_bw_read=PPoly([time_range[0], time_range[1]], [[bw]]),
            capacity=cap,
        )

    @classmethod
    def hdd(
        cls,
        name: str = "HDD",
        bandwidth_MBps: float = 150.0,
        capacity_TB: float = 4.0,
        time_range: Tuple[float, float] = (0, 1e6),
    ) -> StorageTier:
        """Create an HDD tier with typical characteristics."""
        bw = bandwidth_MBps * 1e6
        cap = capacity_TB * 1e12
        return cls(
            name=name,
            tier_index=3,
            I_bw_read=PPoly([time_range[0], time_range[1]], [[bw]]),
            capacity=cap,
        )

    def get_resource_input(self, access_type: AccessType) -> PPoly:
        """Get the bandwidth input function for the specified access type."""
        if access_type == AccessType.READ:
            return self.I_bw_read
        assert self.I_bw_write is not None
        return self.I_bw_write


# =============================================================================
# Environment-Side: Tier Mapping (Cache Hit Rates)
# =============================================================================


@dataclass
class TierMapping:
    """
    Environment-side tier mapping: fraction of accesses served by each tier.

    For dataset k, H^r_{k,j}(p) gives the fraction of reads at progress p
    that are served from tier j. Similarly H^w_{k,j}(p) for writes.

    Constraint: sum_j H_{k,j}(p) = 1 for all p (all accesses go somewhere).

    Attributes:
        dataset_name: Name of the dataset this mapping applies to
        H_read: Dict mapping tier_index -> H^r_{k,j}(p) function
        H_write: Dict mapping tier_index -> H^w_{k,j}(p) function
    """

    dataset_name: str
    H_read: Dict[int, PPoly] = field(default_factory=dict)
    H_write: Dict[int, PPoly] = field(default_factory=dict)

    def validate(self, progress_range: Optional[Tuple[float, float]] = None):
        """Validate that H functions sum to 1 at key points."""
        if not self.H_read:
            return

        # Get progress range from first function
        if progress_range is None:
            first_h = next(iter(self.H_read.values()))
            progress_range = (first_h.x[0], first_h.x[-1])

        # Check at several points
        test_points = [
            progress_range[0],
            (progress_range[0] + progress_range[1]) / 2,
            progress_range[1] - 1e-9,
        ]

        for p in test_points:
            total_read = sum(float(numpy.asarray(h(p)).item()) for h in self.H_read.values())
            if abs(total_read - 1.0) > 1e-6:
                raise ValueError(f"H_read does not sum to 1 at p={p}: sum={total_read}")

            if self.H_write:
                total_write = sum(
                    float(numpy.asarray(h(p)).item()) for h in self.H_write.values()
                )
                if abs(total_write - 1.0) > 1e-6:
                    raise ValueError(
                        f"H_write does not sum to 1 at p={p}: sum={total_write}"
                    )

    @classmethod
    def all_from_tier(
        cls, dataset_name: str, tier_index: int, progress_range: Tuple[float, float]
    ) -> TierMapping:
        """
        Create a mapping where all accesses come from a single tier.

        Useful for:
        - Uncached data (all from disk)
        - Fully cached data (all from memory)
        """
        H = PPoly([progress_range[0], progress_range[1]], [[1.0]])
        return cls(
            dataset_name=dataset_name,
            H_read={tier_index: H},
            H_write={tier_index: H},
        )

    @classmethod
    def cold_then_warm(
        cls,
        dataset_name: str,
        cold_tier: int,
        warm_tier: int,
        warmup_progress: float,
        progress_range: Tuple[float, float],
        warm_hit_rate: float = 1.0,
    ) -> TierMapping:
        """
        Create a two-phase mapping: cold start, then warm cache.

        Args:
            dataset_name: Dataset identifier
            cold_tier: Tier index for cold phase (e.g., disk)
            warm_tier: Tier index for warm phase (e.g., memory)
            warmup_progress: Progress value where cache becomes warm
            progress_range: (start, end) progress values
            warm_hit_rate: Hit rate after warmup (default 1.0 = fully cached)
        """
        p_start, p_end = progress_range

        H_warm = PPoly([p_start, warmup_progress, p_end], [[0.0, warm_hit_rate]])
        H_cold = PPoly([p_start, warmup_progress, p_end], [[1.0, 1.0 - warm_hit_rate]])

        return cls(
            dataset_name=dataset_name,
            H_read={warm_tier: H_warm, cold_tier: H_cold},
            H_write={warm_tier: H_warm, cold_tier: H_cold},
        )

    @classmethod
    def constant_hit_rate(
        cls,
        dataset_name: str,
        cache_tier: int,
        backing_tier: int,
        hit_rate: float,
        progress_range: Tuple[float, float],
    ) -> TierMapping:
        """
        Create a mapping with constant cache hit rate.

        Args:
            dataset_name: Dataset identifier
            cache_tier: Tier index for cache (e.g., memory)
            backing_tier: Tier index for backing store (e.g., disk)
            hit_rate: Fraction of accesses served from cache [0, 1]
            progress_range: (start, end) progress values
        """
        p_start, p_end = progress_range

        H_cache = PPoly([p_start, p_end], [[hit_rate]])
        H_backing = PPoly([p_start, p_end], [[1.0 - hit_rate]])

        return cls(
            dataset_name=dataset_name,
            H_read={cache_tier: H_cache, backing_tier: H_backing},
            H_write={cache_tier: H_cache, backing_tier: H_backing},
        )

    def get_hit_fraction(self, tier_index: int, access_type: AccessType) -> PPoly:
        """Get the hit fraction function for a specific tier."""
        h_dict = self.H_read if access_type == AccessType.READ else self.H_write
        if tier_index in h_dict:
            return h_dict[tier_index]
        else:
            # Not in this tier: return zero
            first_h = next(iter(h_dict.values()))
            return PPoly([first_h.x[0], first_h.x[-1]], [[0.0]])


# =============================================================================
# Cache Behavior Models
# =============================================================================


class CacheBehaviorModel:
    """
    Base class for cache behavior models that compute tier mappings.

    Given process reuse descriptors and environment cache capacities,
    produces H functions for use in TierMapping.
    """

    def compute_tier_mapping(
        self,
        access_profile: LogicalAccessProfile,
        tiers: List[StorageTier],
        progress_range: Tuple[float, float],
    ) -> TierMapping:
        """
        Compute tier mapping from access profile and tier capacities.

        Args:
            access_profile: Logical access requirements for the dataset
            tiers: List of storage tiers (sorted fastest to slowest)
            progress_range: Progress range for the mapping

        Returns:
            TierMapping with H functions for each tier
        """
        raise NotImplementedError("Subclasses must implement compute_tier_mapping")


class DirectHitRateModel(CacheBehaviorModel):
    """
    Option A (simplest): Use directly provided hit-rate function.

    Process provides h_{k,j}(p) = expected hit ratio at tier j.
    Environment accepts it as-is (good when you can profile).
    """

    def __init__(self, hit_rate_func: Callable[[float], float], cache_tier: int = 0):
        """
        Args:
            hit_rate_func: Function p -> hit_rate for the cache tier
            cache_tier: Index of the cache tier (default 0 = fastest)
        """
        self.hit_rate_func = hit_rate_func
        self.cache_tier = cache_tier

    def compute_tier_mapping(
        self,
        access_profile: LogicalAccessProfile,
        tiers: List[StorageTier],
        progress_range: Tuple[float, float],
    ) -> TierMapping:
        """Compute tier mapping using the provided hit rate function."""
        p_start, p_end = progress_range

        num_samples = 100
        step = (p_end - p_start) / num_samples

        x_points = [p_start + i * step for i in range(num_samples + 1)]
        h_values = [self.hit_rate_func(p) for p in x_points[:-1]]

        H_cache = PPoly(x_points, [h_values])
        H_backing = PPoly(x_points, [[1.0 - h for h in h_values]])

        backing_tier = max(t.tier_index for t in tiers)

        return TierMapping(
            dataset_name=access_profile.name,
            H_read={self.cache_tier: H_cache, backing_tier: H_backing},
            H_write={self.cache_tier: H_cache, backing_tier: H_backing},
        )


class WSSModel(CacheBehaviorModel):
    wss_func: Callable[[float], float]
    num_samples: int
    eps: float

    def __init__(
        self,
        wss_func: Callable[[float], float],
        *,
        num_samples: int = 100,
        eps: float = 1e-12,
    ):
        self.wss_func = wss_func
        self.num_samples = num_samples
        self.eps = eps

    def _cumulative_hit_rate(self, wss_bytes: float, capacity_bytes: float) -> float:
        if capacity_bytes <= 0:
            return 0.0
        if wss_bytes <= self.eps:
            return 1.0
        return min(1.0, capacity_bytes / wss_bytes)

    @override
    def compute_tier_mapping(
        self,
        access_profile: LogicalAccessProfile,
        tiers: list[StorageTier],
        progress_range: tuple[float, float],
    ) -> TierMapping:
        p_start, p_end = progress_range
        sorted_tiers = sorted(tiers, key=lambda t: t.tier_index)

        step = (p_end - p_start) / self.num_samples
        x_points = [p_start + i * step for i in range(self.num_samples + 1)]

        h_fractions: dict[int, list[float]] = {t.tier_index: [] for t in sorted_tiers}

        for p in x_points[:-1]:
            wss = max(0.0, self.wss_func(p))
            remaining = 1.0
            prev_cum_hit = 0.0

            for tier in sorted_tiers:
                if tier.capacity is None:
                    h_fractions[tier.tier_index].append(0.0)
                    continue

                cum_hit = self._cumulative_hit_rate(wss, tier.capacity)
                cum_hit = max(prev_cum_hit, cum_hit)
                served = max(0.0, min(remaining, cum_hit - prev_cum_hit))

                h_fractions[tier.tier_index].append(served)
                remaining -= served
                prev_cum_hit = cum_hit

            if sorted_tiers:
                h_fractions[sorted_tiers[-1].tier_index][-1] += remaining

        h_read: dict[int, PPoly] = {}
        for tier_idx, fractions in h_fractions.items():
            if any(f > 0 for f in fractions):
                h_read[tier_idx] = PPoly(x_points, [fractions])

        return TierMapping(
            dataset_name=access_profile.name,
            H_read=h_read,
            H_write=h_read.copy(),
        )

    @classmethod
    def constant(
        cls,
        wss_bytes: float,
        *,
        num_samples: int = 100,
        eps: float = 1e-12,
    ) -> "WSSModel":
        return cls(lambda p: wss_bytes, num_samples=num_samples, eps=eps)

    @classmethod
    def piecewise(
        cls,
        phases: list[tuple[float, float, float]],
        *,
        num_samples: int = 100,
        eps: float = 1e-12,
    ) -> "WSSModel":
        def wss(p: float) -> float:
            for start, end, size in phases:
                if start <= p < end:
                    return size
            return phases[-1][2]

        return cls(wss, num_samples=num_samples, eps=eps)


# =============================================================================
# LRU Eviction Model: Multi-Task Cache Eviction for Sequential Workflows
# =============================================================================


@dataclass
class LRUEvictionModel:
    """
    LRU page-cache eviction model for sequential task chains.

    Models how a fixed-capacity page cache is shared across a sequence of
    tasks that each access a known file.  After task N reads file X, the
    cache holds min(capacity, file_size_X) bytes of X's pages.  The next
    task's hit rate on file Y depends on how much capacity remains after X.

    Core formula (single-file LRU, sequential access):
        remaining = max(0, capacity - previous_file_size)
        hit_rate  = min(1, remaining / current_file_size)

    When the current task accesses the *same* file as the previous task the
    hit rate is min(1, capacity / file_size) since the file's own pages are
    still resident.
    """

    cache_capacity_bytes: float

    def compute_hit_rates(
        self,
        task_sequence: List[Tuple[str, float]],
        *,
        cold_start: bool = True,
    ) -> List[float]:
        """Compute per-task hit rates for a sequential workflow.

        Args:
            task_sequence: Ordered list of (dataset_name, file_size_bytes)
                           for each task in the workflow.
            cold_start: If True the cache is empty before the first task
                        (e.g. after drop_caches).

        Returns:
            List of hit rates, one per task, in [0, 1].
        """
        if not task_sequence:
            return []

        hit_rates: List[float] = []
        cached_dataset: Optional[str] = None
        cached_size: float = 0.0

        for dataset_name, file_size in task_sequence:
            if cold_start and not hit_rates:
                hit_rates.append(0.0)
            elif cached_dataset == dataset_name:
                hr = min(1.0, self.cache_capacity_bytes / file_size) if file_size > 0 else 0.0
                hit_rates.append(hr)
            else:
                remaining = max(0.0, self.cache_capacity_bytes - cached_size)
                hr = min(1.0, remaining / file_size) if file_size > 0 else 0.0
                hit_rates.append(hr)

            cached_dataset = dataset_name
            cached_size = file_size

        return hit_rates

    def compute_tier_mappings(
        self,
        task_sequence: List[Tuple[str, float]],
        cache_tier: int,
        backing_tier: int,
        progress_range: Tuple[float, float],
        *,
        cold_start: bool = True,
    ) -> List[TierMapping]:
        """Compute per-task TierMappings for a sequential workflow.

        Convenience wrapper: calls ``compute_hit_rates`` and wraps
        each result into a ``TierMapping.constant_hit_rate``.

        Args:
            task_sequence: Ordered list of (dataset_name, file_size_bytes).
            cache_tier: Tier index for cache (e.g. 0 = page cache).
            backing_tier: Tier index for backing store (e.g. 1 = disk).
            progress_range: (start, end) progress values for the mappings.
            cold_start: If True the cache is empty before the first task.

        Returns:
            List of TierMapping objects, one per task.
        """
        hit_rates = self.compute_hit_rates(task_sequence, cold_start=cold_start)
        return [
            TierMapping.constant_hit_rate(
                dataset_name=name,
                cache_tier=cache_tier,
                backing_tier=backing_tier,
                hit_rate=hr,
                progress_range=progress_range,
            )
            for (name, _), hr in zip(task_sequence, hit_rates)
        ]


# =============================================================================
# Resource Derivation: Convert (A, Q, H) to Standard BottleMod Resources
# =============================================================================


def derive_tier_resources(
    access_profile: LogicalAccessProfile,
    tier_mapping: TierMapping,
    tiers: List[StorageTier],
) -> Tuple[List[PPoly], List[PPoly]]:
    """
    Derive standard BottleMod resource requirement rate functions from storage hierarchy model.

    For each dataset k and tier j, computes tier-specific resource requirement derivatives:

    Read bandwidth resource:
        R'_{(k,j,bw,r)}(p) = H^r_{k,j}(p) * A'^r_k(p)

    Write bandwidth resource:
        R'_{(k,j,bw,w)}(p) = H^w_{k,j}(p) * A'^w_k(p)

    Returns:
        (requirement_funcs, input_funcs): Lists of PPoly functions ready for BottleMod
            - requirement_funcs: R'_{R,l}(p) for each tier resource (rate functions)
            - input_funcs: I_{R,l}(t) for each tier resource

    The returned functions can be passed directly to Task/TaskExecution.
    """
    requirement_funcs: List[PPoly] = []
    input_funcs: List[PPoly] = []

    A_read_prime = access_profile.A_read.derivative()
    assert access_profile.A_write is not None
    A_write_prime = access_profile.A_write.derivative()

    for tier in tiers:
        H_r = tier_mapping.get_hit_fraction(tier.tier_index, AccessType.READ)

        # R'_{bw,r}(p) = H^r(p) * A'^r(p)
        R_bw_r_prime = H_r * A_read_prime
        requirement_funcs.append(R_bw_r_prime)
        input_funcs.append(tier.I_bw_read)

        # R'_{bw,w}(p) = H^w(p) * A'^w(p)
        H_w = tier_mapping.get_hit_fraction(tier.tier_index, AccessType.WRITE)
        R_bw_w_prime = H_w * A_write_prime
        requirement_funcs.append(R_bw_w_prime)
        assert tier.I_bw_write is not None
        input_funcs.append(tier.I_bw_write)

    return requirement_funcs, input_funcs


def derive_all_tier_resources(
    access_profiles: List[LogicalAccessProfile],
    tier_mappings: List[TierMapping],
    tiers: List[StorageTier],
) -> Tuple[List[PPoly], List[PPoly]]:
    """
    Derive resources for multiple datasets.

    Aggregates resource requirements across all datasets and tiers.
    """
    all_requirements: List[PPoly] = []
    all_inputs: List[PPoly] = []

    for profile, mapping in zip(access_profiles, tier_mappings):
        reqs, inputs = derive_tier_resources(profile, mapping, tiers)
        all_requirements.extend(reqs)
        all_inputs.extend(inputs)

    return all_requirements, all_inputs


# =============================================================================
# Convenience: StorageHierarchyTask wrapping standard Task
# =============================================================================


@dataclass
class StorageHierarchyTask:
    """
    Extended Task that includes storage hierarchy modeling.

    This is a convenience wrapper that:
    1. Takes storage-hierarchy-aware specifications
    2. Derives standard BottleMod resource functions
    3. Creates a standard Task for use with TaskExecution

    Attributes:
        access_profiles: Logical access profiles for each dataset
        tier_mappings: Cache/tier hit rate mappings for each dataset
        tiers: Storage tiers in the environment
        cpu_funcs: Original CPU requirement functions (from standard BottleMod)
        data_funcs: Original data requirement functions (from standard BottleMod)
    """

    access_profiles: List[LogicalAccessProfile]
    tier_mappings: List[TierMapping]
    tiers: List[StorageTier]
    cpu_funcs: List[PPoly] = field(default_factory=list)
    data_funcs: List[Func] = field(default_factory=list)

    _derived_cpu_funcs: List[PPoly] = field(default_factory=list, init=False)
    _derived_input_funcs: List[PPoly] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Derive storage resource functions on initialization."""
        self._derive_storage_resources()

    def _derive_storage_resources(self):
        """Derive storage resource functions from access profiles and tier mappings."""
        storage_reqs, storage_inputs = derive_all_tier_resources(
            self.access_profiles,
            self.tier_mappings,
            self.tiers,
        )

        # Combine with original CPU functions
        self._derived_cpu_funcs = list(self.cpu_funcs) + storage_reqs
        self._derived_input_funcs = storage_inputs

    def to_task(self) -> "Task":
        """
        Convert to standard BottleMod Task.

        Returns:
            Task with CPU functions augmented by storage tier resources
        """
        from bottlemod.task import Task

        return Task(self._derived_cpu_funcs, self.data_funcs)

    def get_storage_input_funcs(self) -> List[PPoly]:
        """Get the derived storage input functions for TaskExecution."""
        return self._derived_input_funcs

    def get_resource_labels(self) -> List[str]:
        """
        Get human-readable labels for each resource.

        Returns:
            List of labels like ["CPU", "DRAM_bw_read", "DRAM_bw_write", ...]
        """
        labels = [f"CPU_{i}" for i in range(len(self.cpu_funcs))]

        for profile in self.access_profiles:
            for tier in self.tiers:
                labels.extend(
                    [
                        f"{tier.name}_{profile.name}_bw_read",
                        f"{tier.name}_{profile.name}_bw_write",
                    ]
                )

        return labels


# =============================================================================
# Utility Functions
# =============================================================================


def get_bottleneck_label(bottleneck_index: int, task: StorageHierarchyTask) -> str:
    """
    Get human-readable bottleneck label.

    Args:
        bottleneck_index: Bottleneck index from TaskExecution (negative = resource)
        task: StorageHierarchyTask with resource labels

    Returns:
        Human-readable label like "data_0" or "DRAM_bw_read"
    """
    if bottleneck_index >= 0:
        return f"data_{bottleneck_index}"
    else:
        resource_idx = -1 - bottleneck_index
        labels = task.get_resource_labels()
        if resource_idx < len(labels):
            return labels[resource_idx]
        else:
            return f"resource_{resource_idx}"


def identify_bottleneck_type(bottleneck_label: str) -> str:
    """
    Classify bottleneck into high-level category.

    Returns one of: "data", "cpu", "disk_bw", "memory_bw"
    """
    label = bottleneck_label.lower()

    if label.startswith("data"):
        return "data"
    elif "cpu" in label:
        return "cpu"
    elif "hdd" in label or "ssd" in label or "nvme" in label or "sata" in label or "disk" in label:
        return "disk_bw"
    elif "dram" in label or "memory" in label or "ram" in label or "pagecache" in label or "page_cache" in label or "cache" in label:
        return "memory_bw"
    else:
        return "unknown"
