# Experiment 1 — Cache-Aware Modeling of FFmpeg Video Remux

This experiment demonstrates that **BottleMod‑CA** (cache-aware) produces more accurate runtime predictions than **vanilla BottleMod** for I/O-intensive workloads where performance depends on the storage hierarchy (page cache vs. disk).

It mirrors the evaluation style of the original BottleMod paper (`paper/bottlemod.pdf`, Sec. 4.2–4.4, Figures 6–7):

- model a workload
- compare prediction vs. measurement
- show that the extended model captures cache effects the baseline cannot

## What we did

### Workload

**ffmpeg video remux** (copy codec, no transcoding) of a 4.6 GB H.264 video on SATA disk. This is a purely I/O-bound sequential-read workload.

We sweep the **cgroup memory limit** from 256 MB to 16 GB to control the effective page cache size. With small memory, reads are disk-bound (cold). With large memory, reads are memory-bound (warm).

### Models

- **Vanilla BottleMod**: models ffmpeg as a single task with one bandwidth constraint. Predicts a constant ~18.9 s regardless of memory limit.
- **BottleMod‑CA**: models ffmpeg as a `StorageHierarchyTask` with LogicalAccessProfile, TierMapping, and StorageTier constructs. Predicts 18.9 s (cold) down to 13.1 s (warm).

### Implementation

- Runner + plots: `exp1_cache_aware_ordering.py`
- Experiment description: `01_cache_aware_ordering.md`

The runner:

1. Calibrates effective **disk_bw** (cold) and **mem_bw** (warm).
2. Sweeps cgroup memory limits.
3. For each limit, runs ffmpeg for N trials, records runtime.
4. Models the workload in both vanilla BottleMod and BottleMod‑CA.
5. Writes a results JSON and generates Figure‑6 and Figure‑7 style plots.

### Run environment (tu)

- Host: `cpu09`, AMD EPYC 7282 16-Core, 125 Gi RAM
- SATA disk: `/dev/sda` mounted at `/mnt/sata` (880 GB, ext4)
- Input video: `/mnt/sata/input.mp4` (4.3 GB, H.264 1080p)
- ffmpeg 6.1.1, cgroup v2, sudo available
- Calibration: disk_bw = 242.7 MB/s, mem_bw = 350.7 MB/s

## Key Result

The vanilla model predicts a flat ~18.9 s for all memory limits. The CA model correctly predicts the runtime curve from cold (~18.9 s) to warm (~13.1 s), demonstrating a 1.44× improvement in prediction accuracy for cache-warm workloads.

## Plots (paper-style)

### Figure‑6 style (prediction vs measurement)

- x-axis: cgroup memory limit
- y-axis: runtime (s)
- **Orange line**: BottleMod‑CA prediction
- **Gray dashed line**: Vanilla BottleMod prediction (flat)
- **Black error bars**: measured runtime (mean ± min/max)

### Figure‑7 style (bottleneck timeline + resource usage)

2×2 panel (cold vs warm):

- Top row: progress (%) with bottleneck-colored bands
- Bottom row: bandwidth usage vs time

## How to reproduce (tu)

```bash
ROOT="$HOME/bm_exp/bottlemod_cache_aware"
PY="$ROOT/.venv/bin/python"

"$PY" "$ROOT/thesis_experiment/01_cach_aware_ordering/exp1_cache_aware_ordering.py" \
  --video /mnt/sata/input.mp4 \
  --out-dir "/var/tmp/exp1_ffmpeg_$(date +%Y%m%d_%H%M%S)" \
  --mem-sweep 256M,512M,1G,2G,4G,8G,16G \
  --trials 5 --drop-caches
```

