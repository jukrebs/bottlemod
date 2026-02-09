# Experiment 1 — Cache-aware ordering (reuse vs eviction)

## Goal

Demonstrate that **workflow ordering** can change whether repeated reads are served from **disk** or from the **page cache**, and that BottleMod‑SH can:

1) predict the bottleneck timeline (disk→memory),
2) identify a performance problem in a naive workflow ordering, and
3) recommend a fix (reordering) that yields a measurable speedup.

This mirrors the original BottleMod paper’s evaluation pattern (Sec. 4.2–4.4): a workflow with a shared bottleneck, a change in allocation/ordering, and a speedup.

## Workflow

We consider a single-machine, buffered I/O workflow consisting of sequential reads:

- Dataset **A** (fits in cache budget **C**)
- Dataset **B** (used to evict A from cache)

Two candidate orderings:

### Baseline (naive)

1. Read A (cold)
2. Read B (cold, evicts cache)
3. Read A again (expected warm, but becomes cold due to eviction)

### Fix (cache-aware ordering)

1. Read A (cold)
2. Read A again (warm, served from page cache)
3. Read B (cold)

## Bottleneck narrative

- Baseline: the **second read of A** becomes **disk-bandwidth bound**.
- Fix: the **second read of A** becomes **memory/page-cache bandwidth bound**, so total runtime decreases.

The insight is *not* “buy a faster disk”; it’s a workflow-level change: **place repeated A access back-to-back** so it is still cached.

## Experimental control

Run the workflow inside a single cgroup (one `systemd-run` scope) with:

- `MemoryMax=C` to bound page cache capacity
- `IOReadBandwidthMax=...` to reduce variance (optional but recommended)

Cold-start each trial using:

- `sync; echo 3 > /proc/sys/vm/drop_caches` (sudo)

## Swept parameter (Figure‑6 style)

Sweep **B read size** as a fraction of the cache budget:

\[
\text{x} = 100 \cdot \frac{|B|}{C} \;\; (\%)
\]

Intuition: for small |B|, A is not evicted → baseline and fix are similar.
For large |B|, A is evicted → baseline is slower and fix wins.

## Measurements

For each x value and for each ordering:

- run N trials
- record per-step wall time and total wall time
- compute mean and min/max (error bars)

Implementation detail:
- use buffered sequential reads (e.g., fio `--rw=read --direct=0`)
- **do not** invalidate caches between steps; only cold-start once per workflow run

## Model / prediction (BottleMod‑SH)

Model the workflow as a single piecewise process over progress p with two datasets (A and B):

- A is read in phases where required
- B is read in phases where required

Tier model:

- Tier 0: memory (page cache) with bandwidth `mem_bw`
- Tier 1: disk with bandwidth `disk_bw`

Tier mapping:

- Baseline: A is served from disk in both reads (cold both times) once B is large enough to evict.
- Fix: the second A read is served from memory.

Use measured `disk_bw` (cold read) and `mem_bw` (warm read) for predictions.

## Plots (match BottleMod paper format)

### Figure‑6 style

For baseline and fix (two stacked panels or two series):

- x-axis: `|B|/C` in %
- y-axis: total workflow time (s)
- **orange solid line**: BottleMod‑SH prediction
- **black min/max bars** (with mean): measured runtime

### Figure‑7 style

2×2 panel for two scenarios (e.g., small |B| and large |B|, or baseline vs fix):

- Top row: progress (%) vs time with full-height **bottleneck-colored bands**
- Bottom row: disk data rate usage vs time (piecewise constant lines)

Use the same general colors/layout as the original paper’s Figure 7.

## Implementation (this repo)

- Runner + plotting: `thesis experiments/exp1_cache_aware_ordering.py`

It produces:
- `exp1_cache_aware_ordering_results.json`
- `fig6_exp1_ABA.png` (baseline)
- `fig6_exp1_AAB.png` (fix)
- `fig7_exp1_baseline_vs_fix.png` (paper Figure‑7 style 2×2 panel)

Typical invocation (run inside a systemd-run scope on `tu`):

```bash
sudo systemd-run --wait --collect \
  --property=CPUAffinity=0-3 \
  --property=MemoryMax=4G --property=MemorySwapMax=0 \
  --property='IOReadBandwidthMax=/dev/sdb2 200M' \
  -- \
  python3 "thesis experiments/exp1_cache_aware_ordering.py" \
    --out-dir /var/tmp/exp1_cache_ordering \
    --cache-bytes 4G --a-bytes 2G \
    --b-bytes-sweep 0,1G,2G,3G,4G,6G,8G \
    --trials 5 --drop-caches
```
