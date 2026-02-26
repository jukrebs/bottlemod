# Experiment 1 — Cache-Aware Modeling of FFmpeg Video Remux

## Goal

Demonstrate that **BottleMod‑CA** (cache-aware) produces more accurate runtime predictions than **vanilla BottleMod** for an I/O-intensive workload whose performance depends on the storage hierarchy.

The experiment uses **ffmpeg video remux** (copy codec, no transcoding) as a real-world sequential-read workload. By varying the available page cache via cgroup memory limits, we control how much of the input file is served from memory vs. disk. The vanilla model cannot capture this effect; the cache-aware model can.

## Workload

**ffmpeg remux** of a 4.6 GB H.264 video file on a SATA disk:

```bash
ffmpeg -y -i /mnt/sata/input.mp4 -c copy /mnt/sata/output.mp4
```

This is a purely I/O-bound workload — CPU utilization during remux is negligible. The bottleneck is sequential read bandwidth, which depends on whether data resides in the page cache or must be read from disk.

### Two-video reordering

When `--video2` is provided, the experiment also runs two sequential remux tasks under a shared cgroup and compares orderings (A→B vs B→A). After task 1, its pages occupy the page cache, reducing cache available for task 2. The model predicts: `remaining_cache = max(0, mem_limit - file_size_task1)`.

- **Video A**: `/mnt/sata/input.mp4` (4.3 GB)
- **Video B**: `/mnt/sata/input_small.mp4` (1020 MB)

Processing the smaller video first leaves more cache room for the larger video.

## Experimental Design

### Swept parameter

**cgroup memory limit** from 256 MB to 16 GB. This controls the effective page cache size:

- **Small memory** (e.g., 256 MB): file cannot be cached → reads are disk-bound → cold performance
- **Large memory** (e.g., 16 GB): file fits in cache → reads are memory-bound → warm performance

### Control

Each trial:

1. Drop caches: `sync; echo 3 > /proc/sys/vm/drop_caches`
2. Run ffmpeg inside a cgroup with the specified `MemoryMax`
3. Record wall-clock runtime

Multiple trials per memory limit for statistical confidence.

### Calibration

Before the sweep, measure:

- **disk_bw**: effective SATA read bandwidth (cold read) → 242.7 MB/s
- **mem_bw**: effective page cache bandwidth (warm read) → 350.7 MB/s

## Models

### Vanilla BottleMod

Models ffmpeg as a single task with:
- One CPU function (minimal compute)
- One data function (sequential read at disk bandwidth)

Produces a **constant** runtime prediction (~18.9 s) regardless of memory limit, because it has no concept of storage tiers.

### BottleMod‑CA

Models ffmpeg as a `StorageHierarchyTask` with:
- **LogicalAccessProfile**: bytes read over progress (sequential read of input file)
- **TierMapping**: cache hit rate function based on cgroup memory limit vs. file size
- **StorageTier** (memory): bandwidth = 350.7 MB/s
- **StorageTier** (disk): bandwidth = 242.7 MB/s

## Plots (BottleMod paper style)

### Figure‑6 style (prediction vs measurement)

- x-axis: cgroup memory limit
- y-axis: runtime (s)
- **Orange line**: BottleMod‑CA prediction
- **Gray dashed line**: Vanilla BottleMod prediction (flat)
- **Black error bars**: measured runtime (mean ± min/max)

### Figure‑7 style (bottleneck timeline)

2×2 panel showing cold vs warm scenarios:

- Top row: progress (%) vs time with bottleneck-colored bands
- Bottom row: resource usage (bandwidth) vs time

### Figure‑8 style (two-video task ordering)

- x-axis: cgroup memory limit
- y-axis: total workflow runtime (s)
- A→B vs B→A prediction lines and measured bars
- Shows that task ordering matters when combined file sizes exceed available cache

### Figure‑9 style (bottleneck timeline for best vs worst ordering)

2×2 panel at the memory limit where ordering difference is largest:

- Left: best ordering (task 1 + task 2 progress)
- Right: worst ordering (task 1 + task 2 progress)
- Bottleneck-colored bands show disk-bound vs memory-bound segments

## Implementation

- Runner + plotting: `exp1_cache_aware_ordering.py`
- Uses `.venv` in project root (patched SciPy)

## Run environment (tu)

- Host: `cpu09`, AMD EPYC 7282 16-Core, 125 Gi RAM
- SATA disk: `/dev/sda` mounted at `/mnt/sata` (880 GB, ext4)
- Input video A: `/mnt/sata/input.mp4` (4.3 GB, H.264 1080p)
- Input video B: `/mnt/sata/input_small.mp4` (1020 MB, H.264)
- ffmpeg 6.1.1, cgroup v2, sudo available

## How to reproduce

```bash
ROOT="$HOME/bm_exp/bottlemod_cache_aware"
PY="$ROOT/.venv/bin/python"

"$PY" "$ROOT/thesis_experiment/01_cach_aware_ordering/exp1_cache_aware_ordering.py" \
  --video /mnt/sata/input.mp4 \
  --video2 /mnt/sata/input_small.mp4 \
  --out-dir "/var/tmp/exp1_ffmpeg_$(date +%Y%m%d_%H%M%S)" \
  --mem-sweep 256M,512M,1G,2G,4G,8G,16G \
  --trials 5 --drop-caches
```
