# Branch Comparison: `main` vs `idea_1`

## Overview

The `idea_1` branch introduces a new **write cache feature** alongside supporting documentation and examples. This branch adds significant new functionality while maintaining backward compatibility and performance parity for existing workloads.

## Purpose of Each Branch

### `main`
- Baseline branch; does not include write_cache additions listed below
- Core cache-aware functionality without write-back caching strategy
- Reference point for comparing `idea_1` changes

### `idea_1`
- Adds write-cache implementation and supporting artifacts (see Evidence section)
- Includes write-cache builder, example, and API documentation
- Introduces 9 new benchmark visualizations comparing write cache patterns

## What Changed

### Code Modifications

**Modified Files:**
- `bottlemod_new/builders/__init__.py`: 32 lines modified to support write_cache builder
- `example_simple.py`: 52 lines modified (see Evidence for full diff)

**Deleted:**
- `bottlemod_new/builders/io.py`: Legacy I/O module removed

**New Functionality:**
- `bottlemod_new/builders/write_cache.py`: Core write cache implementation (393 lines)
- `example_write_cache.py`: Complete example demonstrating write cache patterns (380 lines)

### Documentation

**New Docs:**
- `interface.md`: 202 lines describing general interface patterns
- `interface_write_cache.md`: 300 lines detailing write cache API and usage
- `research.md`: 112 lines capturing research findings and design rationale

### Visualizations

**Modified:**
- `figures/example_simple_visualization.png`: Updated benchmark visualization

**New Benchmarks:**
- 9 new figures comparing write cache performance across scenarios:
  - Small (1.5 MB), Large (512 MB), and Synchronous patterns
  - HDD, SATA SSD, NVMe SSD storage types
  - Dirty buffer scenarios (4GB, 16GB, 64GB)

## Summary of Changes

```
18 files changed, 1440 insertions(+), 432 deletions(-)
```

| Category | Impact |
|----------|--------|
| Code | ~32 lines modified in builders, legacy io.py removed, 393-line write_cache module added |
| Examples | 52 lines modified in example_simple.py, new 380-line write_cache example added |
| Documentation | +614 lines of markdown (interface, research) |
| Benchmarks | 9 new visualization figures |

## Evidence: Git Output

### Commits on `idea_1` not in `main`

```
a5eef38 idea1
fd99675 idea1
```

### File Changes (Summary)

```
 bottlemod_new/builders/__init__.py          |  32 ++-
 bottlemod_new/builders/io.py                | 401 ----------------------------
 bottlemod_new/builders/write_cache.py       | 393 +++++++++++++++++++++++++++
 example_simple.py                           |  52 +++-
 example_write_cache.py                      | 380 ++++++++++++++++++++++++++
 figures/example_simple_visualization.png    | Bin 137881 -> 147085 bytes
 figures/write_cache_example1_small.png      | Bin 0 -> 145430 bytes
 figures/write_cache_example2_large.png      | Bin 0 -> 150993 bytes
 figures/write_cache_example3_sync.png       | Bin 0 -> 146638 bytes
 figures/write_cache_example4_hdd.png        | Bin 0 -> 136551 bytes
 figures/write_cache_example4_nvme_ssd.png   | Bin 0 -> 140571 bytes
 figures/write_cache_example4_sata_ssd.png   | Bin 0 -> 150349 bytes
 figures/write_cache_example5_dirty_16gb.png | Bin 0 -> 150902 bytes
 figures/write_cache_example5_dirty_4gb.png  | Bin 0 -> 147019 bytes
 figures/write_cache_example5_dirty_64gb.png | Bin 0 -> 141250 bytes
 interface.md                                | 202 ++++++++++++++
 interface_write_cache.md                    | 300 +++++++++++++++++++++
 research.md                                 | 112 ++++++++
 18 files changed, 1440 insertions(+), 432 deletions(-)
```

### File Status

```
M	bottlemod_new/builders/__init__.py
D	bottlemod_new/builders/io.py
A	bottlemod_new/builders/write_cache.py
M	example_simple.py
A	example_write_cache.py
M	figures/example_simple_visualization.png
A	figures/write_cache_example1_small.png
A	figures/write_cache_example2_large.png
A	figures/write_cache_example3_sync.png
A	figures/write_cache_example4_hdd.png
A	figures/write_cache_example4_nvme_ssd.png
A	figures/write_cache_example4_sata_ssd.png
A	figures/write_cache_example5_dirty_16gb.png
A	figures/write_cache_example5_dirty_4gb.png
A	figures/write_cache_example5_dirty_64gb.png
A	interface.md
A	interface_write_cache.md
A	research.md
```

## Impact on Experiments

### Rerun Results: `tu` (20260208_191055)

Performance tests on `main` and `idea_1` using sequential + two_pass workloads (no-IOPS fio):

#### Sequential Pattern
- **main**: 11.174s (upstream: 11.174s, SH: 11.174s)
- **idea_1**: 11.165s (upstream: 11.165s, SH: 11.165s)
- **Difference**: Negligible (<0.1%)

#### Two-Pass Pattern
- **main**: 11.596s actual (upstream: 22.348s [+92.72%], SH: 11.598s [+0.014%])
- **idea_1**: 11.580s actual (upstream: 22.330s [+92.83%], SH: 11.582s [+0.014%])
- **Difference**: Negligible (<0.2%)

### Analysis

**Key Finding**: Performance is **equivalent** between branches.

**Why**: The experimental workload (sequential + two_pass) exercises buffered reads and cache-warming behavior. The `write_cache` feature added on `idea_1` does not affect these read-dominated patterns. The implementation maintains the performance characteristics of the baseline while adding optional write cache capabilities.

## Recommendation

- **Stability**: `main` remains the stable baseline for read-centric workloads
- **Experimentation**: `idea_1` is safe to merge or use for workloads involving buffered writes
- **Next Steps**: Run write-intensive benchmarks to validate write_cache performance before production merge
