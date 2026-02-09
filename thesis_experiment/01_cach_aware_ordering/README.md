# Experiment 1 — Cache-aware ordering (A→B→A vs A→A→B)

This experiment demonstrates **why the storage hierarchy / caching addition (BottleMod‑SH)** matters: the *same logical read volume* can be served by **different tiers** (disk vs page cache) depending on **workflow ordering**.

It mirrors the evaluation style of the original BottleMod paper (`paper/bottlemod.pdf`, Sec. 4.2–4.4, Figures 6–7):

- model a workflow
- identify the bottleneck
- apply an actionable fix
- show a speedup
- plot prediction vs measurement in a paper-style format

## What we did

### Workflow

We read two datasets **A** and **B** using buffered sequential reads (Linux page cache enabled).

- **Baseline**: `A → B → A` (second A becomes cold again if B evicts A)
- **Fix**: `A → A → B` (second A immediately reuses cached pages → warm)

### Implementation

- Runner + plots: `exp1_cache_aware_ordering.py`
- Experiment description: `01_cache_aware_ordering.md`

The runner:

1. Calibrates effective **disk_bw** (cold) and **mem_bw** (warm) by reading A twice.
2. Sweeps B size.
3. For each sweep point, runs `ABA` and `AAB` orderings for N trials.
4. Writes a results JSON and generates Figure‑6 and Figure‑7 style plots.

### Run environment (tu)

This run was executed on **`tu`** (`cpu09`) inside a single `systemd-run` scope:

- `MemoryMax=4G` (simulated cache budget)
- `MemorySwapMax=0`
- `CPUAffinity=0-3`
- requested `IOReadBandwidthMax=/dev/sdb2 200M` (variance control)
- `--drop-caches` before each workflow trial (cold start)

The exact output directory on `tu` is recorded in:

- `findings/20260209_094804/out_dir.txt`

## Findings (this run)

Artifacts copied from `tu`:

- `findings/20260209_094804/exp1_cache_aware_ordering_results.json`
- `findings/20260209_094804/fig6_exp1_ABA.png`
- `findings/20260209_094804/fig6_exp1_AAB.png`
- `findings/20260209_094804/fig7_exp1_baseline_vs_fix.png`

Calibration (from the results JSON):

- cold throughput estimate (disk_bw): **~1629 MB/s**
- warm throughput estimate (mem_bw/page cache): **~4994 MB/s**

### Speedup summary

`C = 4 GiB`, `A = 2 GiB`, `trials = 5`, sweep `B ∈ {0,1,2,3,4,6} GiB`.

| B/C (%) | B (GiB) | mean ABA (s) | mean AAB (s) | speedup |
|---:|---:|---:|---:|---:|
| 0 | 0.0 | 2.168 | 2.171 | 1.00× |
| 25 | 1.0 | 2.401 | 2.404 | 1.00× |
| 50 | 2.0 | 4.580 | 3.068 | 1.49× |
| 75 | 3.0 | 5.440 | 3.945 | 1.38× |
| 100 | 4.0 | 6.308 | 4.815 | 1.31× |
| 150 | 6.0 | 8.055 | 6.553 | 1.23× |

Interpretation:

- For small B (≤ 1 GiB), A is not evicted → both orderings behave similarly.
- Once B is large enough to pressure/evict cache, `A→A→B` keeps the second A warm → measurable speedup.
- Speedup shrinks again for very large B because total time becomes dominated by reading B.

## Plots (paper-style)

### Figure‑6 style (prediction vs measurement)

- `fig6_exp1_ABA.png`: baseline (A→B→A)
- `fig6_exp1_AAB.png`: fix (A→A→B)

Style matched to the paper:

- **orange solid prediction line**
- **black min/max error bars** for measured runs

### Figure‑7 style (bottleneck timeline + resource usage)

- `fig7_exp1_baseline_vs_fix.png`

2×2 panel:

- top row: progress (%) with **bottleneck-colored background bands**
- bottom row: disk data rate usage lines
- right column shows y-axis labels/ticks on the right (as in the paper)

## How to reproduce (tu)

Run inside a single scope to keep cache accounting consistent:

```bash
ROOT="$HOME/bm_exp/bottlemod_cache_aware"
PY="$ROOT/.venv/bin/python"

sudo systemd-run --wait --collect \
  --property=CPUAffinity=0-3 \
  --property=MemoryMax=4G --property=MemorySwapMax=0 \
  --property='IOReadBandwidthMax=/dev/sdb2 200M' \
  -- \
  "$PY" "$ROOT/thesis_experiment/01_cach_aware_ordering/exp1_cache_aware_ordering.py" \
    --out-dir "/var/tmp/thesis_exp1_cache_ordering_$(date +%Y%m%d_%H%M%S)" \
    --cache-bytes 4G --a-bytes 2G \
    --b-bytes-sweep 0,1G,2G,3G,4G,6G \
    --trials 5 --drop-caches \
    --data-dir /var/tmp/thesis_exp1_data
```
