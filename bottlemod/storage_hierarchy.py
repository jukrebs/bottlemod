"""
BottleMod-SH: Storage-Hierarchy Extension for BottleMod

This module extends BottleMod to model storage hierarchy effects (caches, tiered storage)
while preserving the original separation of concerns:
- Process descriptors (requirements): time-independent relations over progress p
- Environment descriptors (inputs): time-dependent availability/allocations over real time t

Key abstractions:
- LogicalAccessProfile: Process-side logical access requirements (A^r, A^w, Q^r, Q^w)
- StorageTier: Environment-side tier with bandwidth/IOPS limits
- TierMapping: Environment-side cache/tier hit-rate functions (H^r, H^w)
- CacheBehaviorModel: Computes H from reuse descriptors and cache capacities

After the mapping step, we derive standard BottleMod resource requirement functions R_{R,l}(p)
per tier, enabling reuse of the existing progress calculation algorithm.

References:
- Mattson et al. 1970: Stack distance / reuse distance theory
- BottleMod (ICPE'25): Modeling Data Flows and Tasks for Fast Bottleneck Analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, cast

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
    IOPS = auto()  # operations/s


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
        Q_read: Q^r_k(p) - cumulative logical read operations up to progress p
        Q_write: Q^w_k(p) - cumulative logical write operations up to progress p

    The ratio A'/Q' gives the average request size, distinguishing sequential
    (large requests, low Q') from random (small requests, high Q') access.

    Example - Sequential streaming reads:
        A_read = PPoly([0, 100], [[1000]])  # 1000 bytes per progress unit
        Q_read = PPoly([0, 100], [[1]])     # 1 op per progress unit (1000B requests)

    Example - Random 4KiB reads:
        A_read = PPoly([0, 100], [[4096]])  # 4096 bytes per progress unit
        Q_read = PPoly([0, 100], [[1]])     # 1 op per progress unit (4KiB requests)
    """

    name: str
    A_read: PPoly
    A_write: Optional[PPoly] = None
    Q_read: Optional[PPoly] = None
    Q_write: Optional[PPoly] = None

    def __post_init__(self):
        """Validate and set defaults for access profile."""
        # Default Q from A assuming large sequential requests (1MB)
        if self.Q_read is None and self.A_read is not None:
            # Assume 1MB request size by default (sequential)
            self.Q_read = self.A_read * (1.0 / (1024 * 1024))

        if self.Q_write is None and self.A_write is not None:
            self.Q_write = self.A_write * (1.0 / (1024 * 1024))

        # Ensure writes default to zero if not specified
        if self.A_write is None:
            self.A_write = PPoly([self.A_read.x[0], self.A_read.x[-1]], [[0]])
        if self.Q_write is None:
            self.Q_write = PPoly([self.A_read.x[0], self.A_read.x[-1]], [[0]])

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
            LogicalAccessProfile with linear A^r and low Q^r (large requests)
        """
        bytes_per_progress = total_bytes / max_progress
        ops_per_progress = bytes_per_progress / (1024 * 1024)

        A_read = PPoly([0, max_progress], [[bytes_per_progress], [0]])
        Q_read = PPoly([0, max_progress], [[ops_per_progress], [0]])

        return cls(
            name=name,
            A_read=A_read,
            Q_read=Q_read,
        )

    @classmethod
    def random_read(
        cls,
        name: str,
        total_bytes: float,
        max_progress: float,
        request_size: float = 4096,
    ) -> LogicalAccessProfile:
        """
        Create a profile for random reads (e.g., database lookups).

        Args:
            name: Dataset identifier
            total_bytes: Total bytes to read
            max_progress: Maximum progress value
            request_size: Size of each read request (default 4KiB)

        Returns:
            LogicalAccessProfile with matching A^r and Q^r (small requests)
        """
        bytes_per_progress = total_bytes / max_progress
        ops_per_progress = bytes_per_progress / request_size

        A_read = PPoly([0, max_progress], [[bytes_per_progress], [0]])
        Q_read = PPoly([0, max_progress], [[ops_per_progress], [0]])

        return cls(
            name=name,
            A_read=A_read,
            Q_read=Q_read,
        )

    @classmethod
    def piecewise(
        cls, name: str, phases: List[Tuple[float, float, float, float]]
    ) -> LogicalAccessProfile:
        """
        Create a piecewise profile for multi-phase workloads.

        Args:
            name: Dataset identifier
            phases: List of (start_p, end_p, bytes_rate, ops_rate) tuples
                   Each phase has constant rates for bytes and ops per progress unit.

        Returns:
            LogicalAccessProfile with piecewise linear A^r and Q^r
        """
        if not phases:
            raise ValueError("At least one phase required")

        # Build breakpoints and coefficients
        x_points = [phases[0][0]]
        a_coeffs = []
        q_coeffs = []

        for start_p, end_p, bytes_rate, ops_rate in phases:
            x_points.append(end_p)
            a_coeffs.append([bytes_rate])
            q_coeffs.append([ops_rate])

        # Integrate rates to get cumulative functions
        # PPoly stores derivative coefficients, so we need cumulative
        A_read = PPoly(x_points, [a_coeffs[i] for i in range(len(a_coeffs))])
        A_read = A_read.antiderivative()
        # Shift to start at 0
        offset = A_read(x_points[0])
        A_read = A_read - offset

        Q_read = PPoly(x_points, [q_coeffs[i] for i in range(len(q_coeffs))])
        Q_read = Q_read.antiderivative()
        offset = Q_read(x_points[0])
        Q_read = Q_read - offset

        return cls(name=name, A_read=A_read, Q_read=Q_read)

    def get_derivative(self, access_type: AccessType) -> PPoly:
        """Get the derivative (rate) function for the specified access type."""
        if access_type == AccessType.READ:
            return self.A_read.derivative()
        else:
            assert self.A_write is not None
            return self.A_write.derivative()

    def get_ops_derivative(self, access_type: AccessType) -> PPoly:
        """Get the operations rate function for the specified access type."""
        if access_type == AccessType.READ:
            assert self.Q_read is not None
            return self.Q_read.derivative()
        else:
            assert self.Q_write is not None
            return self.Q_write.derivative()


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
        I_iops_read: I^{iops,r}_j(t) - read IOPS available (ops/s)
        I_iops_write: I^{iops,w}_j(t) - write IOPS available (ops/s)
        capacity: Effective capacity in bytes (for cache hit rate calculation)
    """

    name: str
    tier_index: int
    I_bw_read: PPoly
    I_bw_write: Optional[PPoly] = None
    I_iops_read: Optional[PPoly] = None
    I_iops_write: Optional[PPoly] = None
    capacity: Optional[float] = None

    def __post_init__(self):
        """Set defaults for unspecified resource functions."""
        time_range = [self.I_bw_read.x[0], self.I_bw_read.x[-1]]

        if self.I_bw_write is None:
            # Default write BW = read BW
            self.I_bw_write = PPoly(time_range, self.I_bw_read.c.copy())

        if self.I_iops_read is None:
            # Default: no IOPS limit (very high)
            self.I_iops_read = PPoly(time_range, [[1e12]])

        if self.I_iops_write is None:
            self.I_iops_write = PPoly(time_range, [[1e12]])

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
        iops_k: float = 500.0,
        capacity_GB: float = 500.0,
        time_range: Tuple[float, float] = (0, 1e6),
    ) -> StorageTier:
        """Create an NVMe SSD tier with typical characteristics."""
        bw = bandwidth_GBps * 1e9
        iops = iops_k * 1e3
        cap = capacity_GB * 1e9
        return cls(
            name=name,
            tier_index=1,
            I_bw_read=PPoly([time_range[0], time_range[1]], [[bw]]),
            I_iops_read=PPoly([time_range[0], time_range[1]], [[iops]]),
            capacity=cap,
        )

    @classmethod
    def sata_ssd(
        cls,
        name: str = "SATA_SSD",
        bandwidth_MBps: float = 500.0,
        iops_k: float = 100.0,
        capacity_GB: float = 1000.0,
        time_range: Tuple[float, float] = (0, 1e6),
    ) -> StorageTier:
        """Create a SATA SSD tier with typical characteristics."""
        bw = bandwidth_MBps * 1e6
        iops = iops_k * 1e3
        cap = capacity_GB * 1e9
        return cls(
            name=name,
            tier_index=2,
            I_bw_read=PPoly([time_range[0], time_range[1]], [[bw]]),
            I_iops_read=PPoly([time_range[0], time_range[1]], [[iops]]),
            capacity=cap,
        )

    @classmethod
    def hdd(
        cls,
        name: str = "HDD",
        bandwidth_MBps: float = 150.0,
        iops: float = 150.0,
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
            I_iops_read=PPoly([time_range[0], time_range[1]], [[iops]]),
            capacity=cap,
        )

    def get_resource_input(
        self, resource_type: ResourceType, access_type: AccessType
    ) -> PPoly:
        """Get the resource input function for the specified resource and access type."""
        if resource_type == ResourceType.BANDWIDTH:
            if access_type == AccessType.READ:
                return self.I_bw_read
            assert self.I_bw_write is not None
            return self.I_bw_write
        else:
            if access_type == AccessType.READ:
                assert self.I_iops_read is not None
                return self.I_iops_read
            assert self.I_iops_write is not None
            return self.I_iops_write


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
            total_read = sum(cast(float, h(p)) for h in self.H_read.values())
            if abs(total_read - 1.0) > 1e-6:
                raise ValueError(f"H_read does not sum to 1 at p={p}: sum={total_read}")

            if self.H_write:
                total_write = sum(cast(float, h(p)) for h in self.H_write.values())
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


class StackDistanceModel(CacheBehaviorModel):
    """
    Option B (recommended): Stack-distance / reuse-distance model.

    Process provides a stack-distance CDF F_k(p, s):
        Probability that an access at progress p has reuse distance <= s bytes

    Environment provides cache capacity C_j, and computes:
        hit_{k,j}(p) = F_k(p, C_j)

    For an inclusive hierarchy, converts hits to tier fractions:
        H^r_{k,0}(p) = hit_{k,0}(p)
        H^r_{k,1}(p) = (1 - hit_{k,0}(p)) * hit_{k,1}(p)
        ...
        H^r_{k,J}(p) = product_{u=0}^{J-1} (1 - hit_{k,u}(p))

    References:
        - Mattson et al. 1970: "Evaluation techniques for storage hierarchies"
        - Ding & Zhong 2001: "Reuse Distance Analysis"
    """

    def __init__(
        self,
        stack_distance_cdf: Callable[[float, float], float],
        working_set_size: Optional[float] = None,
    ):
        """
        Args:
            stack_distance_cdf: Function (p, s) -> probability of reuse distance <= s
            working_set_size: Total working set size in bytes (for normalization)
        """
        self.stack_distance_cdf = stack_distance_cdf
        self.working_set_size = working_set_size

    def compute_tier_mapping(
        self,
        access_profile: LogicalAccessProfile,
        tiers: List[StorageTier],
        progress_range: Tuple[float, float],
    ) -> TierMapping:
        """Compute tier mapping using stack distance CDF and tier capacities."""
        p_start, p_end = progress_range

        # Sort tiers by index (fastest first)
        sorted_tiers = sorted(tiers, key=lambda t: t.tier_index)

        # Sample progress points
        num_samples = 100
        step = (p_end - p_start) / num_samples
        x_points = [p_start + i * step for i in range(num_samples + 1)]

        # Compute hit rates at each tier for each progress point
        # H[tier_idx][sample_idx] = fraction of accesses served from tier
        H_fractions: Dict[int, List[float]] = {t.tier_index: [] for t in sorted_tiers}

        for p in x_points[:-1]:  # Don't include last point (used as boundary)
            remaining = 1.0  # Fraction not yet served by faster tiers

            for tier in sorted_tiers:
                if tier.capacity is None:
                    # No capacity = no caching at this tier, skip
                    H_fractions[tier.tier_index].append(0.0)
                    continue

                # Hit rate at this tier given remaining misses from faster tiers
                hit_rate = self.stack_distance_cdf(p, tier.capacity)
                fraction_served = remaining * hit_rate
                H_fractions[tier.tier_index].append(fraction_served)
                remaining *= 1.0 - hit_rate

            # Last tier gets all remaining (backing store)
            if sorted_tiers:
                last_tier = sorted_tiers[-1]
                # Adjust last tier to include remaining
                H_fractions[last_tier.tier_index][-1] += remaining

        H_read: Dict[int, PPoly] = {}
        for tier_idx, fractions in H_fractions.items():
            if any(f > 0 for f in fractions):
                H_read[tier_idx] = PPoly(x_points, [fractions])

        return TierMapping(
            dataset_name=access_profile.name,
            H_read=H_read,
            H_write=H_read.copy(),  # Same for writes by default
        )

    @classmethod
    def uniform_reuse(cls, working_set_size: float) -> StackDistanceModel:
        """
        Create a model for uniform random access over a working set.

        With uniform access to W bytes, the probability of reuse distance <= s
        is approximately min(1, s/W) for LRU caches.

        Args:
            working_set_size: Size of the working set in bytes

        Returns:
            StackDistanceModel with uniform reuse pattern
        """

        def uniform_cdf(p: float, s: float) -> float:
            return min(1.0, s / working_set_size)

        return cls(uniform_cdf, working_set_size)

    @classmethod
    def streaming_no_reuse(cls) -> StackDistanceModel:
        """
        Create a model for streaming access with no reuse.

        Each byte is accessed exactly once, so reuse distance is infinite.
        Cache hit rate is 0 regardless of cache size.
        """

        def no_reuse_cdf(p: float, s: float) -> float:
            return 0.0

        return cls(no_reuse_cdf)

    @classmethod
    def full_reuse(cls) -> StackDistanceModel:
        """
        Create a model for workloads that fully reuse all data.

        After initial cold miss, all accesses hit in cache (if capacity sufficient).
        """

        def full_reuse_cdf(p: float, s: float) -> float:
            return 1.0  # Effectively infinite cache or very small working set

        return cls(full_reuse_cdf)


class PhaseBasedCacheModel(CacheBehaviorModel):
    """
    Piecewise cache model for multi-phase workloads.

    Explicitly captures cold -> warm transitions via phases:
    - Phase 1 (first pass / warmup): low hit rate
    - Phase 2 (steady-state reuse): high hit rate
    - Phase 3 (optional, thrash): hit drops when working set grows

    This matches BottleMod's piecewise "events" philosophy.
    """

    @dataclass
    class Phase:
        """A phase with specific cache behavior."""

        start_progress: float
        end_progress: float
        hit_rate: float  # Constant hit rate during this phase

    def __init__(self, phases: List[Phase], cache_tier: int = 0):
        """
        Args:
            phases: List of phases defining hit rate over progress
            cache_tier: Index of the cache tier
        """
        self.phases = sorted(phases, key=lambda p: p.start_progress)
        self.cache_tier = cache_tier

    def compute_tier_mapping(
        self,
        access_profile: LogicalAccessProfile,
        tiers: List[StorageTier],
        progress_range: Tuple[float, float],
    ) -> TierMapping:
        """Compute piecewise tier mapping from phases."""
        # Build x points and coefficients from phases
        x_points = []
        h_values = []

        for phase in self.phases:
            x_points.append(phase.start_progress)
            h_values.append(phase.hit_rate)

        # Add final point
        x_points.append(self.phases[-1].end_progress)

        H_cache = PPoly(x_points, [h_values])
        H_backing = PPoly(x_points, [[1.0 - value for value in h_values]])

        # Find backing tier
        backing_tier = max(t.tier_index for t in tiers)

        return TierMapping(
            dataset_name=access_profile.name,
            H_read={self.cache_tier: H_cache, backing_tier: H_backing},
            H_write={self.cache_tier: H_cache, backing_tier: H_backing},
        )


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

    Read IOPS resource:
        R'_{(k,j,iops,r)}(p) = H^r_{k,j}(p) * Q'^r_k(p)

    Similarly for writes.

    Returns:
        (requirement_funcs, input_funcs): Lists of PPoly functions ready for BottleMod
            - requirement_funcs: R'_{R,l}(p) for each tier resource (rate functions)
            - input_funcs: I_{R,l}(t) for each tier resource

    The returned functions can be passed directly to Task/TaskExecution.
    """
    requirement_funcs: List[PPoly] = []
    input_funcs: List[PPoly] = []

    # Get access rate derivatives
    A_read_prime = access_profile.A_read.derivative()
    assert access_profile.Q_read is not None
    assert access_profile.A_write is not None
    assert access_profile.Q_write is not None
    Q_read_prime = access_profile.Q_read.derivative()
    A_write_prime = access_profile.A_write.derivative()
    Q_write_prime = access_profile.Q_write.derivative()

    for tier in tiers:
        # Read bandwidth resource for this tier
        H_r = tier_mapping.get_hit_fraction(tier.tier_index, AccessType.READ)

        # R'_{bw,r}(p) = H^r(p) * A'^r(p)
        R_bw_r_prime = H_r * A_read_prime
        requirement_funcs.append(R_bw_r_prime)
        input_funcs.append(tier.I_bw_read)

        # Read IOPS resource for this tier (if IOPS is limiting)
        R_iops_r_prime = H_r * Q_read_prime
        requirement_funcs.append(R_iops_r_prime)
        assert tier.I_iops_read is not None
        input_funcs.append(tier.I_iops_read)

        # Write bandwidth resource
        H_w = tier_mapping.get_hit_fraction(tier.tier_index, AccessType.WRITE)

        R_bw_w_prime = H_w * A_write_prime
        requirement_funcs.append(R_bw_w_prime)
        assert tier.I_bw_write is not None
        input_funcs.append(tier.I_bw_write)

        # Write IOPS resource
        R_iops_w_prime = H_w * Q_write_prime
        requirement_funcs.append(R_iops_w_prime)
        assert tier.I_iops_write is not None
        input_funcs.append(tier.I_iops_write)

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
            List of labels like ["CPU", "DRAM_bw_read", "DRAM_iops_read", ...]
        """
        labels = [f"CPU_{i}" for i in range(len(self.cpu_funcs))]

        for profile in self.access_profiles:
            for tier in self.tiers:
                labels.extend(
                    [
                        f"{tier.name}_{profile.name}_bw_read",
                        f"{tier.name}_{profile.name}_iops_read",
                        f"{tier.name}_{profile.name}_bw_write",
                        f"{tier.name}_{profile.name}_iops_write",
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

    Returns one of: "data", "cpu", "disk_bw", "disk_iops", "memory_bw", "memory_iops"
    """
    label = bottleneck_label.lower()

    if label.startswith("data"):
        return "data"
    elif "cpu" in label:
        return "cpu"
    elif "hdd" in label or "ssd" in label or "nvme" in label or "sata" in label:
        if "iops" in label:
            return "disk_iops"
        else:
            return "disk_bw"
    elif "dram" in label or "memory" in label or "ram" in label:
        if "iops" in label:
            return "memory_iops"
        else:
            return "memory_bw"
    else:
        return "unknown"
