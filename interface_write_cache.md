# BottleMod Write Cache Interface

This document explains how to model **write-back caching** (Linux page cache for writes) using BottleMod's PPoly framework.

---

## Overview

Linux uses **write-back caching** for file writes:
1. Writes go to the **page cache** (memory) at memory bandwidth
2. Dirty pages accumulate until the **dirty limit** is reached
3. When dirty limit is hit, writes are **throttled** to disk speed
4. Background flushing (`pdflush`/`kworker`) syncs dirty pages to disk

This creates a **two-phase write behavior** that BottleMod models.

---

## Linux Dirty Page Parameters

Linux controls write caching via these kernel parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vm.dirty_ratio` | 20% | Hard limit - writes block when dirty pages exceed this |
| `vm.dirty_background_ratio` | 10% | Background flush starts at this threshold |
| `vm.dirty_bytes` | 0 | Absolute dirty limit (overrides ratio if non-zero) |
| `vm.dirty_background_bytes` | 0 | Absolute background threshold |

**Example:** On a 32 GB system with defaults:
- Background flush starts at 3.2 GB dirty pages
- Writes block at 6.4 GB dirty pages

---

## Mathematical Model

### Phase 1: Burst Phase (dirty buffer filling)

For time $t \in [0, t_{transition}]$ where $t_{transition} = \frac{\text{dirty\_limit}}{\text{cache\_bandwidth}}$:

- **Effective bandwidth**: $B_{cache}$ (cache speed)
- **Cumulative I/O**: $I(t) = B_{cache} \cdot t$

All writes absorbed by cache at cache bandwidth.

### Phase 2: Disk-Limited Phase (dirty limit reached)

For time $t > t_{transition}$:

- **Effective bandwidth**: $B_{disk}$ (disk speed)
- **Cumulative I/O**: $I(t) = \text{dirty\_limit} + B_{disk} \cdot (t - t_{transition})$

Writes limited by disk bandwidth as dirty pages are recycled.

### Total Time

$$T_{total} = \frac{\text{dirty\_limit}}{B_{cache}} + \frac{\text{write\_total} - \text{dirty\_limit}}{B_{disk}}$$

---

## PPoly Implementation

The model uses a single "write" resource with a **piecewise bandwidth function** in the environment.

### Task Requirements

```python
from bottlemod_new.builders.write_cache import create_write_task

# Large write (100 GB) with 16 GB dirty limit
task = create_write_task(
    write_total=100e9,      # 100 GB to write
    dirty_limit=16e9,       # 16 GB dirty page limit (used for reference)
)
```

This creates a linear write requirement:

```python
# Write requirement: R(p) = write_total * p
write_func = PPoly([0, 1.0], [[100e9], [0]])
```

### Environment with Bandwidth Transition

```python
from bottlemod_new.builders.write_cache import create_write_environment

# Environment with dirty limit causing bandwidth transition
env = create_write_environment(
    cache_bandwidth=25e9,   # 25 GB/s memory
    disk_bandwidth=500e6,   # 500 MB/s disk
    dirty_limit=16e9,       # 16 GB dirty limit
)
```

This creates a piecewise cumulative input function:

```python
# Time when dirty_limit is reached: t = 16e9 / 25e9 = 0.64s
# Phase 1: I(t) = 25e9 * t  for t ∈ [0, 0.64]
# Phase 2: I(t) = 16e9 + 500e6 * (t - 0.64) for t > 0.64

write_input = PPoly(
    [0, 0.64, inf],
    [[25e9, 500e6], [0, 16e9 - 500e6*0.64]]
)
```

---

## Worked Example

### Scenario

- **Write size**: 100 GB
- **Dirty limit**: 16 GB
- **Cache bandwidth**: 25 GB/s
- **Disk bandwidth**: 500 MB/s

### Analysis

**Transition time**: $t_{transition} = \frac{16\text{ GB}}{25\text{ GB/s}} = 0.64\text{ s}$

**Phase 1** (time 0 → 0.64s):
- Data written: 16 GB
- Bandwidth: 25 GB/s (cache)
- Progress: 16%

**Phase 2** (time 0.64s → end):
- Data to write: 84 GB
- Bandwidth: 500 MB/s (disk)
- Time: $\frac{84\text{ GB}}{0.5\text{ GB/s}} = 168\text{ s}$

**Total time**: $0.64 + 168 = 168.64$ seconds

### Bottleneck Analysis

| Time (s) | Progress | Phase | Bottleneck |
|----------|----------|-------|------------|
| 0 - 0.64 | 0% - 16% | Cache | Disk (at cache speed) |
| 0.64 - 168.64 | 16% - 100% | Disk | Disk (at disk speed) |

---

## Synchronous Writes (O_SYNC / O_DIRECT)

Some applications bypass the page cache:
- Database transaction logs (O_SYNC)
- Direct I/O applications (O_DIRECT)
- Applications calling fsync() after each write

For these cases, use:

```python
from bottlemod_new.builders.write_cache import (
    create_sync_write_task,
    create_sync_write_environment,
)

# All writes go directly to disk
task = create_sync_write_task(write_total=10e9)
env = create_sync_write_environment(disk_bandwidth=500e6)
```

This models writes at disk speed with no cache benefit.

---

## Builder Functions Reference

### `create_write_task()`

```python
def create_write_task(
    write_total: float,           # Total bytes to write
    dirty_limit: float,           # Dirty page limit (bytes) - for reference
    cpu_requirement: float = None, # Optional CPU work
    cpu_name: str = "CPU_0",
    disk_name: str = "Disk",
    max_progress: float = 1.0,
) -> TaskRequirements
```

### `create_write_environment()`

```python
def create_write_environment(
    cache_bandwidth: float,       # Cache write speed (bytes/s)
    disk_bandwidth: float = None, # Disk write speed (bytes/s)
    dirty_limit: float = None,    # Dirty limit for bandwidth transition
    cpu_bandwidth: float = None,  # Optional CPU speed
    cpu_name: str = "CPU_0",
    disk_name: str = "Disk",
) -> ExecutionEnvironment
```

When `dirty_limit` and `disk_bandwidth` are both provided, the environment creates a piecewise bandwidth function that transitions from cache to disk speed.

### `create_sync_write_task()` / `create_sync_write_environment()`

For O_SYNC/O_DIRECT writes that bypass cache.

---

## Typical Hardware Bandwidths

| Component | Bandwidth | Notes |
|-----------|-----------|-------|
| DDR4-3200 | ~25 GB/s | Single channel |
| DDR5-4800 | ~38 GB/s | Single channel |
| NVMe SSD | 3-7 GB/s | Gen4/Gen5 |
| SATA SSD | 500 MB/s | SATA III limit |
| HDD (7200 RPM) | 150 MB/s | Sequential writes |

---

## Use Cases

### 1. Log File Writing
```python
# Continuous logging: small writes, likely all cached
task = create_write_task(write_total=1e9, dirty_limit=4e9)
env = create_write_environment(cache_bandwidth=25e9)  # No disk transition
```

### 2. Large File Copy
```python
# 50 GB file copy: exceeds typical dirty limit
task = create_write_task(write_total=50e9, dirty_limit=6.4e9)
env = create_write_environment(
    cache_bandwidth=25e9,
    disk_bandwidth=500e6,
    dirty_limit=6.4e9,
)
```

### 3. Database Checkpoint
```python
# Database needs durability: use sync writes
task = create_sync_write_task(write_total=10e9)
env = create_sync_write_environment(disk_bandwidth=500e6)
```

### 4. Video Rendering
```python
# Large output file with CPU work
task = create_write_task(
    write_total=100e9,
    dirty_limit=16e9,
    cpu_requirement=1e12,  # 1 TFLOP of computation
)
env = create_write_environment(
    cache_bandwidth=25e9,
    disk_bandwidth=3e9,     # NVMe SSD
    dirty_limit=16e9,
    cpu_bandwidth=100e9,    # 100 GFLOP/s
)
```

---

## Example Output

Running `example_write_cache.py` produces:

```
Example 1: Small Write (fits in cache)
  Write size: 10 GB, Dirty limit: 16 GB
  Completion time: 0.40 seconds (cache speed only)

Example 2: Large Write (exceeds cache)  
  Write size: 100 GB, Dirty limit: 16 GB
  Completion time: 168.64 seconds
    Phase 1: 0.64s at cache speed (25 GB/s)
    Phase 2: 168s at disk speed (500 MB/s)

Example 3: Synchronous Write (O_SYNC)
  Write size: 10 GB
  Completion time: 20.00 seconds (disk speed only)
  Slowdown vs cached: 50x
```

---

## Limitations & Future Work

Current model limitations:
1. **No IOPS modeling**: Assumes sequential writes (no small file overhead)
2. **Simplified flush model**: Uses hard transition at dirty_limit
3. **No memory pressure**: Assumes dirty limit is constant
4. **No I/O scheduler**: Ignores scheduling overhead

Potential extensions:
- Mixed read/write workloads
- Random vs sequential access patterns  
- IOPS-limited scenarios (many small files)
- Memory pressure effects on dirty limit
- Background flush modeling (dirty_background_limit)
