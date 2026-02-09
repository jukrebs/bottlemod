"""
BottleMod-SH Worked Examples

Demonstrates the three key scenarios from the theoretical model:
1. Sequential scan once (no reuse) -> disk bottleneck
2. Two-pass analytics (cold then warm) -> cold-to-warm transition
3. Random 4KiB reads -> IOPS bottleneck dominates

Each example shows:
- Model parameterization
- Progress prediction
- Bottleneck identification
- Comparison with original BottleMod (no storage hierarchy)
"""

from bottlemod.func import Func
from bottlemod.ppoly import PPoly
from bottlemod.task import Task, TaskExecution
from bottlemod.storage_hierarchy import (
    LogicalAccessProfile,
    StorageTier,
    TierMapping,
    StackDistanceModel,
    PhaseBasedCacheModel,
    derive_tier_resources,
    StorageHierarchyTask,
    get_bottleneck_label,
    identify_bottleneck_type,
)


def _filter_zero_requirements(
    requirements: list[PPoly], inputs: list[PPoly]
) -> tuple[list[PPoly], list[PPoly]]:
    filtered_requirements = []
    filtered_inputs = []
    for requirement, input_func in zip(requirements, inputs):
        if all(cc == 0 for cc in requirement.c.flatten()):
            continue
        filtered_requirements.append(requirement)
        filtered_inputs.append(input_func)
    return filtered_requirements, filtered_inputs


def _make_available_data_input(max_progress: float, time_range: tuple[float, float]) -> Func:
    return Func([time_range[0], time_range[1]], [[0, max_progress]])


def example_1_sequential_scan():
    """
    Example 1: Sequential scan once (no reuse) -> disk bottleneck
    
    Scenario:
    - Read a file of size S once, sequentially
    - No reuse, so H_disk(p) = 1, H_mem(p) = 0
    - Progress p = output bytes (expansion factor alpha = 1)
    
    Expected behavior:
    - Disk bandwidth bottleneck
    - Runtime scales linearly with file size
    - Max progress speed = B / alpha where B = disk bandwidth
    """
    print("=" * 60)
    print("Example 1: Sequential Scan (No Reuse)")
    print("=" * 60)
    
    FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10 GB
    MAX_PROGRESS = FILE_SIZE  # Progress = bytes processed
    TIME_RANGE = (0, 1000)  # seconds
    
    DISK_BW = 500 * 1024 * 1024  # 500 MB/s (SATA SSD)
    MEM_BW = 25 * 1024 * 1024 * 1024  # 25 GB/s (DDR4)
    
    access_profile = LogicalAccessProfile.sequential_read(
        name="input_file",
        total_bytes=FILE_SIZE,
        max_progress=MAX_PROGRESS,
    )
    
    tiers = [
        StorageTier.memory(name="DRAM", bandwidth_GBps=25.0, time_range=TIME_RANGE),
        StorageTier.sata_ssd(name="SSD", bandwidth_MBps=500.0, time_range=TIME_RANGE),
    ]
    
    tier_mapping = TierMapping.all_from_tier(
        dataset_name="input_file",
        tier_index=2,  # All from SSD (no cache hit)
        progress_range=(0, MAX_PROGRESS),
    )
    
    storage_reqs, storage_inputs = derive_tier_resources(
        access_profile, tier_mapping, tiers
    )
    storage_reqs, storage_inputs = _filter_zero_requirements(
        storage_reqs, storage_inputs
    )
    
    cpu_cycles_per_byte = 100
    cpu_func = PPoly([0, MAX_PROGRESS], [[cpu_cycles_per_byte]])
    cpu_input = PPoly([TIME_RANGE[0], TIME_RANGE[1]], [[1e12]])  # 1 THz (not limiting)
    
    all_cpu_funcs = [cpu_func] + storage_reqs
    all_cpu_inputs = [cpu_input] + storage_inputs
    
    data_func = Func([0, FILE_SIZE], [[1, 0]])
    data_input = _make_available_data_input(FILE_SIZE, TIME_RANGE)
    
    task = Task(all_cpu_funcs, [data_func])
    execution = TaskExecution(task, all_cpu_inputs, [data_input])
    
    progress_func, bottlenecks = execution.get_result()
    
    final_time = progress_func.x[-1]
    final_progress = progress_func(final_time)
    
    print(f"File size: {FILE_SIZE / 1e9:.1f} GB")
    print(f"Disk bandwidth: {DISK_BW / 1e6:.0f} MB/s")
    print(f"Predicted runtime: {final_time:.2f} seconds")
    print(f"Expected runtime (FILE_SIZE / BW): {FILE_SIZE / DISK_BW:.2f} seconds")
    print(f"Final progress: {final_progress / 1e9:.1f} GB")
    
    print("\nBottleneck analysis:")
    for i, bn in enumerate(set(bottlenecks)):
        if bn < 0:
            print(f"  Resource {-1 - bn}: storage tier resource")
        else:
            print(f"  Data {bn}: data availability")
    
    return progress_func, bottlenecks


def example_2_two_pass_cold_warm():
    """
    Example 2: Two-pass analytics (cold then warm) -> cold-to-warm transition
    
    Scenario:
    - Read same file twice
    - First pass: data loaded from disk (cold cache)
    - Second pass: data served from memory (warm cache)
    
    Case 1: S << page cache capacity -> second pass from memory
    Case 2: S >> capacity -> both passes from disk
    
    Expected behavior:
    - Piecewise bottleneck shift from disk to memory/CPU
    - Second run faster in Case 1
    """
    print("\n" + "=" * 60)
    print("Example 2: Two-Pass Analytics (Cold -> Warm)")
    print("=" * 60)
    
    FILE_SIZE = 4 * 1024 * 1024 * 1024  # 4 GB (fits in 16GB cache)
    MAX_PROGRESS = 2 * FILE_SIZE  # Two passes
    WARMUP_PROGRESS = FILE_SIZE  # After first pass, cache is warm
    TIME_RANGE = (0, 100)
    
    DISK_BW = 500 * 1024 * 1024  # 500 MB/s
    MEM_BW = 25 * 1024 * 1024 * 1024  # 25 GB/s
    
    access_profile = LogicalAccessProfile.sequential_read(
        name="dataset",
        total_bytes=2 * FILE_SIZE,  # Total bytes read across both passes
        max_progress=MAX_PROGRESS,
    )
    
    tiers = [
        StorageTier.memory(name="DRAM", bandwidth_GBps=25.0, capacity_GB=16.0, time_range=TIME_RANGE),
        StorageTier.sata_ssd(name="SSD", bandwidth_MBps=500.0, time_range=TIME_RANGE),
    ]
    
    tier_mapping = TierMapping.cold_then_warm(
        dataset_name="dataset",
        cold_tier=2,  # SSD
        warm_tier=0,  # DRAM
        warmup_progress=WARMUP_PROGRESS,
        progress_range=(0, MAX_PROGRESS),
        warm_hit_rate=0.999,  # Avoid zero-rate segments
    )
    
    storage_reqs, storage_inputs = derive_tier_resources(
        access_profile, tier_mapping, tiers
    )
    storage_reqs, storage_inputs = _filter_zero_requirements(
        storage_reqs, storage_inputs
    )
    
    cpu_func = PPoly([0, MAX_PROGRESS], [[100]])  # 100 cycles/byte
    cpu_input = PPoly([TIME_RANGE[0], TIME_RANGE[1]], [[1e12]])
    
    all_cpu_funcs = [cpu_func] + storage_reqs
    all_cpu_inputs = [cpu_input] + storage_inputs
    
    data_func = Func([0, 2 * FILE_SIZE], [[1, 0]])
    data_input = _make_available_data_input(2 * FILE_SIZE, TIME_RANGE)
    
    task = Task(all_cpu_funcs, [data_func])
    execution = TaskExecution(task, all_cpu_inputs, [data_input])
    
    progress_func, bottlenecks = execution.get_result()
    
    time_at_warmup = None
    for x in progress_func.x:
        if progress_func(x) >= WARMUP_PROGRESS:
            time_at_warmup = x
            break
    
    final_time = progress_func.x[-1]
    
    print(f"File size: {FILE_SIZE / 1e9:.1f} GB")
    print(f"Cache capacity: 16 GB (file fits)")
    print(f"\nFirst pass (cold):")
    print(f"  Progress 0 -> {WARMUP_PROGRESS / 1e9:.1f} GB")
    if time_at_warmup:
        print(f"  Time: 0 -> {time_at_warmup:.2f} seconds")
        print(f"  Expected (disk BW limited): {FILE_SIZE / DISK_BW:.2f} seconds")
    
    print(f"\nSecond pass (warm):")
    print(f"  Progress {WARMUP_PROGRESS / 1e9:.1f} -> {MAX_PROGRESS / 1e9:.1f} GB")
    if time_at_warmup:
        second_pass_time = final_time - time_at_warmup
        print(f"  Time: {time_at_warmup:.2f} -> {final_time:.2f} seconds ({second_pass_time:.2f}s)")
        print(f"  Expected (memory BW limited): {FILE_SIZE / MEM_BW:.4f} seconds")
        print(f"  Speedup ratio: {(FILE_SIZE / DISK_BW) / second_pass_time:.1f}x")
    
    return progress_func, bottlenecks


def example_3_random_iops():
    """
    Example 3: Random 4KiB reads -> IOPS bottleneck dominates
    
    Scenario:
    - Random 4KiB reads over a working set W
    - If W < cache capacity: memory-limited (fast)
    - If W > cache capacity: disk IOPS-limited (slow)
    
    Expected behavior:
    - IOPS constraint binds before bandwidth
    - Bottleneck identified as "disk iops"
    """
    print("\n" + "=" * 60)
    print("Example 3: Random 4KiB Reads (IOPS Bottleneck)")
    print("=" * 60)
    
    WORKING_SET = 100 * 1024 * 1024 * 1024  # 100 GB (larger than cache)
    TOTAL_READS = 1_000_000  # 1 million random reads
    REQUEST_SIZE = 4096  # 4 KiB
    TOTAL_BYTES = TOTAL_READS * REQUEST_SIZE
    MAX_PROGRESS = TOTAL_BYTES
    TIME_RANGE = (0, 10000)
    
    DISK_BW = 500 * 1024 * 1024  # 500 MB/s
    DISK_IOPS = 100_000  # 100K IOPS (typical SSD)
    CACHE_CAPACITY = 16 * 1024 * 1024 * 1024  # 16 GB
    
    access_profile = LogicalAccessProfile.random_read(
        name="random_dataset",
        total_bytes=TOTAL_BYTES,
        max_progress=MAX_PROGRESS,
        request_size=REQUEST_SIZE,
    )
    
    tiers = [
        StorageTier.memory(name="DRAM", bandwidth_GBps=25.0, capacity_GB=16.0, time_range=TIME_RANGE),
        StorageTier(
            name="SSD",
            tier_index=1,
            I_bw_read=PPoly([TIME_RANGE[0], TIME_RANGE[1]], [[DISK_BW]]),
            I_iops_read=PPoly([TIME_RANGE[0], TIME_RANGE[1]], [[DISK_IOPS]]),
            capacity=1e15,  # Large disk
        ),
    ]
    
    cache_model = StackDistanceModel.uniform_reuse(WORKING_SET)
    tier_mapping = cache_model.compute_tier_mapping(
        access_profile,
        tiers,
        (0, MAX_PROGRESS),
    )
    
    storage_reqs, storage_inputs = derive_tier_resources(
        access_profile, tier_mapping, tiers
    )
    storage_reqs, storage_inputs = _filter_zero_requirements(
        storage_reqs, storage_inputs
    )
    
    cpu_func = PPoly([0, MAX_PROGRESS], [[10]])  # 10 cycles/byte
    cpu_input = PPoly([TIME_RANGE[0], TIME_RANGE[1]], [[1e12]])
    
    all_cpu_funcs = [cpu_func] + storage_reqs
    all_cpu_inputs = [cpu_input] + storage_inputs
    
    data_func = Func([0, TOTAL_BYTES], [[1, 0]])
    data_input = _make_available_data_input(TOTAL_BYTES, TIME_RANGE)
    
    task = Task(all_cpu_funcs, [data_func])
    execution = TaskExecution(task, all_cpu_inputs, [data_input])
    
    progress_func, bottlenecks = execution.get_result()
    
    final_time = progress_func.x[-1]
    
    print(f"Working set size: {WORKING_SET / 1e9:.0f} GB")
    print(f"Cache capacity: {CACHE_CAPACITY / 1e9:.0f} GB")
    print(f"Total reads: {TOTAL_READS:,}")
    print(f"Request size: {REQUEST_SIZE} bytes")
    print(f"Total bytes: {TOTAL_BYTES / 1e9:.2f} GB")
    print(f"\nDisk specs:")
    print(f"  Bandwidth: {DISK_BW / 1e6:.0f} MB/s")
    print(f"  IOPS: {DISK_IOPS:,}")
    
    hit_rate = CACHE_CAPACITY / WORKING_SET
    print(f"\nExpected cache hit rate: {hit_rate:.1%}")
    
    disk_reads = TOTAL_READS * (1 - hit_rate)
    expected_time_bw = (disk_reads * REQUEST_SIZE) / DISK_BW
    expected_time_iops = disk_reads / DISK_IOPS
    
    print(f"\nRuntime predictions:")
    print(f"  If BW-limited: {expected_time_bw:.2f} seconds")
    print(f"  If IOPS-limited: {expected_time_iops:.2f} seconds")
    print(f"  Actual (min): {max(expected_time_bw, expected_time_iops):.2f} seconds")
    print(f"  Predicted by model: {final_time:.2f} seconds")
    
    print(f"\nBottleneck: {'IOPS' if expected_time_iops > expected_time_bw else 'Bandwidth'}")
    
    return progress_func, bottlenecks


def compare_with_original_bottlemod():
    """
    Showcase comparison: Original BottleMod vs BottleMod-SH
    
    Demonstrates accuracy gained in two scenarios:
    1. Cache-warm speedup (Example 2)
    2. Random IOPS bottleneck (Example 3)
    """
    print("\n" + "=" * 60)
    print("Comparison: Original BottleMod vs BottleMod-SH")
    print("=" * 60)
    
    print("\n--- Scenario 1: Two-Pass Cache Warmup ---")
    print("\nOriginal BottleMod:")
    print("  - Cannot represent 'second pass faster' without manual intervention")
    print("  - No cache state or tier concept")
    print("  - Would predict: both passes at same speed (disk-limited)")
    
    print("\nBottleMod-SH:")
    print("  - H(p) captures cold->warm transition via piecewise definition")
    print("  - First pass: H_disk=1, H_mem=0 (disk bottleneck)")
    print("  - Second pass: H_disk=0, H_mem=1 (memory bottleneck)")
    print("  - Correctly predicts speedup and identifies bottleneck shift")
    
    print("\n--- Scenario 2: Random IOPS Bottleneck ---")
    print("\nOriginal BottleMod:")
    print("  - Only models bandwidth-like resources")
    print("  - Would predict runtime based on bandwidth alone")
    print("  - Systematic misprediction for random I/O workloads")
    
    print("\nBottleMod-SH:")
    print("  - Separate Q(p) tracks operation count")
    print("  - I^iops(t) constrains operations per second")
    print("  - Correctly identifies when IOPS binds before bandwidth")
    print("  - Bottleneck label: 'disk_iops' vs 'disk_bw'")


if __name__ == "__main__":
    print("BottleMod-SH Storage Hierarchy Examples\n")
    
    try:
        example_1_sequential_scan()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_2_two_pass_cold_warm()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        example_3_random_iops()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    compare_with_original_bottlemod()
    
    print("\n" + "=" * 60)
    print("Examples complete.")
    print("=" * 60)
