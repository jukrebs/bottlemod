#!/usr/bin/env python3
"""
Write Cache Example - BottleMod

Demonstrates write-back cache modeling in BottleMod.

Linux uses write-back caching:
- Writes go to page cache (memory) at cache speed
- Dirty pages are flushed to disk asynchronously in the background
- Application sees cache speed until dirty limit is reached
- After dirty limit, writes are throttled to disk speed

This example shows three scenarios:
1. Small write (fits in dirty buffer) - all at cache speed
2. Large write (exceeds dirty buffer) - two-phase: cache then disk
3. Synchronous write (O_SYNC) - always at disk speed
"""

from pathlib import Path

from bottlemod_new import (
    TaskExecution,
    plot_combined_analysis,
    print_bottleneck_summary,
    print_bottleneck_timeline,
)
from bottlemod_new.builders.write_cache import (
    create_sync_write_environment,
    create_sync_write_task,
    create_write_environment,
    create_write_task,
)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def example1_small_write(visualize=False):
    """
    Example 1: Small write that fits in dirty buffer.

    Scenario:
    - Write 10 GB of data
    - Dirty limit: 16 GB
    - Cache bandwidth: 25 GB/s (DDR4)
    - Disk bandwidth: 500 MB/s (SATA SSD)

    Result: All writes go to cache at 25 GB/s
    Expected time: 10 GB / 25 GB/s = 0.4 seconds
    """
    print("\n" + "=" * 70)
    print("Example 1: Small Write (fits in cache)")
    print("=" * 70)

    # Task: 10 GB write, 16 GB dirty limit
    task = create_write_task(
        write_total=10e9,  # 10 GB
        dirty_limit=16e9,  # 16 GB (all writes fit)
    )

    # Environment: DDR4 memory only (no disk needed since all fits in cache)
    env = create_write_environment(
        cache_bandwidth=25e9,  # 25 GB/s
        # disk_bandwidth not needed - all writes fit in cache
    )

    # Execute analysis
    execution = TaskExecution(task, env)

    # Print results
    print("\nWrite size: 10 GB")
    print("Dirty limit: 16 GB")
    print("Cache bandwidth: 25 GB/s")
    print("Disk bandwidth: 500 MB/s")
    print("\nAll writes fit in cache -> cache speed only")
    print_bottleneck_timeline(execution)
    print_bottleneck_summary(execution)

    completion_time = execution.progress_function.solve(1.0)[0]
    print(f"\nCompletion time: {completion_time:.4f} seconds")
    print(f"Theoretical (cache only): {10e9 / 25e9:.4f} seconds")

    if visualize:
        print("\nGenerating visualization...")
        fig = plot_combined_analysis(execution, title="Small Write (Fits in Cache)")
        out_path = FIGURES_DIR / "write_cache_example1_small.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return execution


def example2_large_write(visualize=False):
    """
    Example 2: Large write exceeding dirty buffer.

    Scenario:
    - Write 100 GB of data
    - Dirty limit: 16 GB
    - Cache bandwidth: 25 GB/s (DDR4)
    - Disk bandwidth: 500 MB/s (SATA SSD)

    Result: Two-phase execution
    - Phase 1 (0-16%): Write at cache speed (25 GB/s)
    - Phase 2 (16-100%): Write at disk speed (500 MB/s)

    Expected time:
    - Phase 1: 16 GB / 25 GB/s = 0.64 seconds
    - Phase 2: 84 GB / 500 MB/s = 168 seconds
    - Total: ~168.64 seconds
    """
    print("\n" + "=" * 70)
    print("Example 2: Large Write (exceeds cache)")
    print("=" * 70)

    # Task: 100 GB write, 16 GB dirty limit
    task = create_write_task(
        write_total=100e9,  # 100 GB
        dirty_limit=16e9,  # 16 GB (transition at 16%)
    )

    # Environment: DDR4 memory + SATA SSD with dirty limit
    env = create_write_environment(
        cache_bandwidth=25e9,  # 25 GB/s
        disk_bandwidth=500e6,  # 500 MB/s
        dirty_limit=16e9,  # 16 GB dirty limit
    )

    # Execute analysis
    execution = TaskExecution(task, env)

    # Print results
    print("\nWrite size: 100 GB")
    print("Dirty limit: 16 GB")
    print(f"Transition point: {16 / 100 * 100:.0f}% progress")
    print("Cache bandwidth: 25 GB/s")
    print("Disk bandwidth: 500 MB/s")
    print("\nPhase 1 (0-16%): Cache speed")
    print("Phase 2 (16-100%): Disk speed (bottleneck!)")
    print_bottleneck_timeline(execution)
    print_bottleneck_summary(execution)

    completion_time = execution.progress_function.solve(1.0)[0]
    print(f"\nCompletion time: {completion_time:.2f} seconds")
    print(f"Theoretical phase 1 (cache): {16e9 / 25e9:.2f} seconds")
    print(f"Theoretical phase 2 (disk): {84e9 / 500e6:.2f} seconds")

    if visualize:
        print("\nGenerating visualization...")
        fig = plot_combined_analysis(execution, title="Large Write (Exceeds Cache)")
        out_path = FIGURES_DIR / "write_cache_example2_large.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return execution


def example3_sync_write(visualize=False):
    """
    Example 3: Synchronous write (O_SYNC/O_DIRECT).

    Some applications bypass the page cache:
    - Databases using O_SYNC for durability
    - Direct I/O (O_DIRECT) for predictable latency
    - fsync() after every write

    Scenario:
    - Write 10 GB of data (same as example 1)
    - Disk bandwidth: 500 MB/s (SATA SSD)

    Result: All writes at disk speed (no cache benefit)
    Expected time: 10 GB / 500 MB/s = 20 seconds
    """
    print("\n" + "=" * 70)
    print("Example 3: Synchronous Write (O_SYNC)")
    print("=" * 70)

    # Task: 10 GB synchronous write
    task = create_sync_write_task(write_total=10e9)

    # Environment: SATA SSD only (cache not used)
    env = create_sync_write_environment(disk_bandwidth=500e6)

    # Execute analysis
    execution = TaskExecution(task, env)

    # Print results
    print("\nWrite size: 10 GB (same as example 1)")
    print("Mode: O_SYNC (synchronous)")
    print("Disk bandwidth: 500 MB/s")
    print("\nNo cache benefit -> disk speed only")
    print_bottleneck_timeline(execution)
    print_bottleneck_summary(execution)

    completion_time = execution.progress_function.solve(1.0)[0]
    print(f"\nCompletion time: {completion_time:.2f} seconds")
    print(f"Comparison with cached write (example 1): {0.4:.2f} seconds")
    print(f"Slowdown factor: {completion_time / 0.4:.1f}x")

    if visualize:
        print("\nGenerating visualization...")
        fig = plot_combined_analysis(execution, title="Synchronous Write (O_SYNC)")
        out_path = FIGURES_DIR / "write_cache_example3_sync.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return execution


def example4_nvme_vs_hdd(visualize=False):
    """
    Example 4: Impact of storage speed on large writes.

    Compares large write performance on different storage:
    - NVMe SSD: 3 GB/s
    - SATA SSD: 500 MB/s
    - HDD: 150 MB/s

    Scenario:
    - Write 100 GB of data
    - Dirty limit: 16 GB
    - Cache bandwidth: 25 GB/s (DDR4)
    """
    print("\n" + "=" * 70)
    print("Example 4: Storage Speed Comparison")
    print("=" * 70)

    write_total = 100e9
    dirty_limit = 16e9
    cache_bandwidth = 25e9

    storage_configs = [
        ("NVMe SSD", 3e9),
        ("SATA SSD", 500e6),
        ("HDD", 150e6),
    ]

    results = []

    for storage_name, disk_bandwidth in storage_configs:
        task = create_write_task(write_total=write_total, dirty_limit=dirty_limit)
        env = create_write_environment(
            cache_bandwidth=cache_bandwidth,
            disk_bandwidth=disk_bandwidth,
            dirty_limit=dirty_limit,
        )
        execution = TaskExecution(task, env)
        completion_time = execution.progress_function.solve(1.0)[0]
        results.append((storage_name, disk_bandwidth, completion_time, execution))

    print("\nWrite size: 100 GB")
    print("Dirty limit: 16 GB")
    print("Cache bandwidth: 25 GB/s")
    print(f"\n{'Storage':<12} {'Bandwidth':<12} {'Time':<12} {'Speedup':<12}")
    print("-" * 48)

    baseline_time = results[-1][2]  # HDD as baseline
    for storage_name, bandwidth, time, _ in results:
        speedup = baseline_time / time
        print(
            f"{storage_name:<12} {bandwidth / 1e9:.2f} GB/s     {time:>8.2f}s    {speedup:>6.1f}x"
        )

    if visualize:
        print("\nGenerating visualizations...")
        for storage_name, _, _, execution in results:
            fig = plot_combined_analysis(
                execution, title=f"100 GB Write on {storage_name}"
            )
            out_path = (
                FIGURES_DIR
                / f"write_cache_example4_{storage_name.lower().replace(' ', '_')}.png"
            )
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {out_path}")

    return results


def example5_dirty_limit_tuning(visualize=False):
    """
    Example 5: Impact of dirty limit on write performance.

    Shows how tuning vm.dirty_bytes affects write performance.

    Scenario:
    - Write 100 GB of data
    - Cache bandwidth: 25 GB/s
    - Disk bandwidth: 500 MB/s
    - Dirty limits: 4 GB, 16 GB, 64 GB
    """
    print("\n" + "=" * 70)
    print("Example 5: Dirty Limit Tuning")
    print("=" * 70)

    write_total = 100e9
    cache_bandwidth = 25e9
    disk_bandwidth = 500e6

    dirty_limits = [4e9, 16e9, 64e9]

    results = []

    for dirty_limit in dirty_limits:
        task = create_write_task(write_total=write_total, dirty_limit=dirty_limit)
        env = create_write_environment(
            cache_bandwidth=cache_bandwidth,
            disk_bandwidth=disk_bandwidth,
            dirty_limit=dirty_limit,
        )
        execution = TaskExecution(task, env)
        completion_time = execution.progress_function.solve(1.0)[0]

        # Calculate time at cache speed
        cache_phase = min(dirty_limit, write_total)
        disk_phase = max(0, write_total - dirty_limit)
        cache_time = cache_phase / cache_bandwidth
        disk_time = disk_phase / disk_bandwidth

        results.append(
            {
                "dirty_limit": dirty_limit,
                "completion_time": completion_time,
                "cache_time": cache_time,
                "disk_time": disk_time,
                "execution": execution,
            }
        )

    print("\nWrite size: 100 GB")
    print("Cache bandwidth: 25 GB/s")
    print("Disk bandwidth: 500 MB/s")
    print(
        f"\n{'Dirty Limit':<14} {'Cache Phase':<14} {'Disk Phase':<14} {'Total Time':<14}"
    )
    print("-" * 56)

    for r in results:
        dirty_gb = r["dirty_limit"] / 1e9
        print(
            f"{dirty_gb:>8.0f} GB     {r['cache_time']:>8.2f}s      {r['disk_time']:>8.2f}s      {r['completion_time']:>8.2f}s"
        )

    print("\nNote: Larger dirty limit = more time at cache speed")
    print(
        "Trade-off: Higher dirty limit increases memory pressure and data loss risk on crash"
    )

    if visualize:
        print("\nGenerating visualizations...")
        for r in results:
            dirty_gb = r["dirty_limit"] / 1e9
            fig = plot_combined_analysis(
                r["execution"], title=f"100 GB Write with {dirty_gb:.0f} GB Dirty Limit"
            )
            out_path = FIGURES_DIR / f"write_cache_example5_dirty_{int(dirty_gb)}gb.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {out_path}")

    return results


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  BottleMod - Write Cache Examples")
    print("#" * 70)
    print("\nLinux Write-Back Cache Model:")
    print("- Writes go to page cache at memory speed")
    print("- When dirty limit reached, writes throttled to disk speed")
    print("- This creates two-phase write behavior for large writes")

    example1_small_write(visualize=True)
    example2_large_write(visualize=True)
    example3_sync_write(visualize=True)
    example4_nvme_vs_hdd(visualize=True)
    example5_dirty_limit_tuning(visualize=True)

    print("\n" + "#" * 70)
    print("#  All examples completed!")
    print("#" * 70)
