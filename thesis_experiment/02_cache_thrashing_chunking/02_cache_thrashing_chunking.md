# Experiment 2 — Thrashing and the “chunking” fix

## Goal

Demonstrate a workload where caching does **not** help (thrash), and show a workflow-level fix (chunking/reordering) that restores cache benefit.

This complements Experiment 1 by showing BottleMod‑SH can also predict *when* caching will fail and what change restores performance.

## Workflow

Two-pass analytics on a dataset D (sequential read twice), but the second pass is **not warm** because the working set exceeds cache or because the workflow interleaves other reads.

### Baseline

- Read D once (cold)
- Interleave with additional reads so that |working set| > cache capacity C
- Read D again → thrash → disk-bound again

### Fix

- Process D in chunks such that chunk size ≤ C
- For each chunk: read chunk once (cold) then reuse it immediately (warm)

## Bottleneck narrative

- Baseline: disk bandwidth bottleneck dominates both passes.
- Fix: second access within each chunk becomes memory-bound; overall wall time drops.

## What to vary

- cache budget C (MemoryMax)
- chunk size
- total D size

## What to show

- runtime vs chunk size (prediction + measurement)
- bottleneck timeline showing frequent disk→mem transitions (fixed) vs disk-only (baseline)

## Implementation pointers (repo)

- Model primitives: `bottlemod/storage_hierarchy.py` (TierMapping, StackDistanceModel, PhaseBasedCacheModel)
- Worked reference: `bottlemod/examples_storage_hierarchy.py` (two-pass example)
