# Experiment 1 — Cache-Aware Task Ordering for FFmpeg Video Remux

This experiment demonstrates that **BottleMod‑CA** (cache-aware) correctly predicts the runtime impact of task ordering when two files compete for a shared page cache, while **vanilla BottleMod** cannot.

## Core Claim

When two files A and B share a cgroup-limited page cache that can hold one file but not both (`file_size < mem_limit < 2 × file_size`), the order in which tasks process them determines cache utilization:

- **Interleaved** (A-B-A-B): each file evicts the other → mostly cold reads
- **Grouped** (A-A-B-B): second pass reuses the first pass's cached pages → partially warm reads

BottleMod‑CA predicts this ordering effect. Vanilla BottleMod sees no difference.

## Experiment Design

### Files and Memory

- **File A**: `/mnt/sata/input_2g_a.mp4` — 1.9 GB (first 15800s of input.mp4)
- **File B**: `/mnt/sata/input_2g_b.mp4` — 1.9 GB (second 15800s of input.mp4)
- **Memory limit**: 3 GB (cgroup v2)

With 1.9 GB files and 3 GB memory: one file fits, but not both. After processing one file, `remaining_cache = max(0, 3.0 − 1.9) = 1.1 GB`, giving the next different-file task a `hit_rate = 1.1 / 1.9 ≈ 0.60`. A same-file follow-up gets `hit_rate = 1.0`.

### Workflow (4 sequential remux tasks)

Each ordering runs 4 `ffmpeg -i <file> -c copy <out>` remux operations (I/O-bound, no transcoding) inside a **single persistent cgroup scope** so page cache charges carry over between tasks.

| Task | Interleaved order | Hit rate | Grouped order | Hit rate |
|------|-------------------|----------|---------------|----------|
| 1    | Op1(A)            | 0.00     | Op1(A)        | 0.00     |
| 2    | Op1(B)            | 0.60     | Op2(A)        | 1.00     |
| 3    | Op2(A)            | 0.60     | Op1(B)        | 0.60     |
| 4    | Op2(B)            | 0.60     | Op2(B)        | 1.00     |

### Models

- **Vanilla BottleMod**: single-tier bandwidth model. Predicts the same runtime for both orderings.
- **BottleMod‑CA**: two-tier `StorageHierarchyTask` with cache eviction tracking. Predicts faster runtime for grouped ordering due to higher effective hit rates. Uses a weighted-harmonic-mean bandwidth model for serial page access: `eff_bw = 1 / (hit/mem_bw + (1−hit)/disk_bw)`.

### Implementation

- **Current experiment**: `exp1_reordering.py` — 4-task workflow, persistent cgroup, interleaved vs grouped
- **Legacy sweep findings**: produced with now-removed `exp1_cache_aware_ordering.py`
- **No-eviction memory-increase validation**: `../02_memory_increase/`
- Experiment description: `01_cache_aware_ordering.md`

## Key Result

| Model | Interleaved (s) | Grouped (s) | Speedup |
|-------|------------------|-------------|---------|
| **BottleMod‑CA** | 39.46 | 35.43 | **1.11×** |
| **Measured** | 37.07 ± 2.05 | 33.87 ± 1.42 | **1.09×** |
| **Vanilla BottleMod** | 48.63 | 48.63 | 1.00× |

**The CA model predicts a 1.11× speedup from cache-friendly ordering; the measured speedup is 1.09×.** Vanilla BottleMod predicts no difference between orderings.

### 10× scale result (18.7 GB × 2 files, 30 GB memory)

| Model | Interleaved (s) | Grouped (s) | Speedup |
|-------|------------------|-------------|---------|
| **BottleMod‑CA** | 379.69 | 359.20 | **1.06×** |
| **Measured** | 379.72 ± 7.69 | 339.31 ± 0.87 | **1.12×** |
| **Vanilla BottleMod** | 426.24 | 426.24 | 1.00× |

The ordering effect remains visible at 10× file size. Grouped order is still clearly faster than interleaved in measured runs.

### Per-task accuracy

| Task condition | CA prediction | Measured |
|----------------|---------------|----------|
| Cold (hit=0%) | ~12 s | ~10–12 s |
| Partial (hit=60%) | ~9 s | ~8–10 s |
| Warm (hit=100%) | ~7 s | ~7–8 s |

## Plots

### Workflow detail (bottleneck timeline + resource usage)

Two 2×N panels (one per ordering) following `paper_figures_eval.py` style:

- **Top row**: progress (%) with bottleneck-colored bands (orange = disk-bound, blue = memory-bound)
- **Bottom row**: bandwidth usage vs time

The grouped plot clearly shows alternating disk-bound and memory-bound segments (Op2 tasks benefit from cache). The interleaved plot is mostly disk-bound throughout.

### Summary comparison (bar chart)

Grouped bar chart showing total workflow runtime for each (model × ordering) combination with error bars on measured data.

### Per-task breakdown (stacked bars)

Per-task runtimes stacked by task, comparing CA vs vanilla vs measured for both orderings.

## Run environment (tu)

- Host: `cpu09`, AMD EPYC 7282 16-Core, 125 Gi RAM
- SATA disk: `/dev/sda` mounted at `/mnt/sata` (880 GB, ext4)
- File A: `/mnt/sata/input_2g_a.mp4` (1.9 GB, H.264, inode 14)
- File B: `/mnt/sata/input_2g_b.mp4` (1.9 GB, H.264, inode 16)
- ffmpeg 6.1.1, cgroup v2, sudo available
- Calibration (run 20260226_171444): disk_bw = 165.4 MB/s, mem_bw = 283.8 MB/s

## How to reproduce (tu)

```bash
ROOT="$HOME/bm_exp/bottlemod_cache_aware"
PY="$ROOT/.venv/bin/python"

PYTHONPATH="$ROOT" "$PY" \
  "$ROOT/thesis_experiment/01_cach_aware_ordering/exp1_reordering.py" \
  --video-a /mnt/sata/input_2g_a.mp4 \
  --video-b /mnt/sata/input_2g_b.mp4 \
  --mem-limit 3G --trials 5 --drop-caches \
  --out-dir "/var/tmp/exp1_reordering_$(date +%Y%m%d_%H%M%S)"
```

## Findings

| Timestamp | Script | Files | Memory | Notes |
|-----------|--------|-------|--------|-------|
| `20260226_181215` | `exp1_reordering.py` | 18.7 GB × 2 | 30G | 10× reordering run: grouped remains faster than interleaved |
| `20260226_171444` | `exp1_reordering.py` | 1.9 GB × 2 | 3 GB | **Best result** — persistent cgroup |
| `20260226_124156` | `exp1_cache_aware_ordering.py` | 4.3 GB + 1020 MB | sweep | Two-video memory sweep (different-sized files) |
| `20260220_123431` | `exp1_cache_aware_ordering.py` | 4.3 GB | sweep | Single-video baseline |

## Flow chart

- See `bottlemod_ca_flowchart.md` for a Mermaid flow chart of the BottleMod-CA pipeline used in these experiments.

## Technical notes

### Persistent cgroup (critical)

Each ordering's 4 tasks must run inside a **single `systemd-run --scope`** cgroup, not individual `systemd-run --wait` invocations. When a transient cgroup is destroyed between tasks, its page cache charges are released to the global pool, and the next task's fresh cgroup benefits from warm global cache regardless of ordering — defeating the experiment.

### CA runtime modeling

`exp1_reordering.py` now uses the original BottleMod-CA `TaskExecution` runtime directly for prediction (same style as the legacy sweep script), while still using the same eviction-based hit-rate setup and two-tier storage hierarchy.
