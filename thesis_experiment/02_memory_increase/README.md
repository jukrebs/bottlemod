# Experiment 01 — Memory Increase (No-Eviction Validation)

This companion experiment validates that the ordering speedup is caused by cache eviction pressure.

## Goal

Run the same four-task remux workflow in two memory regimes:

- **Eviction regime**: `file_size < mem_limit < file_a + file_b`
- **No-eviction regime**: `mem_limit >= file_a + file_b`

If the speedup is truly cache-driven, grouped ordering should be faster only in the eviction regime and collapse to ~1.0× in the no-eviction regime.

## Script

- `exp2_memory_increase_no_eviction.py`

The script uses `thesis_experiment/01_cach_aware_ordering/exp1_reordering.py` for shared modeling and measurement helpers.

## Results

### Base scale (1.9 GB × 2 files)

- Eviction (`3G`): measured speedup `1.075×`
- No eviction (`5G`): measured speedup `0.969×` (≈ no ordering benefit)

### 10× scale (18.7 GB × 2 files)

- Eviction (`30G`): measured speedup `1.150×`
- No eviction (`70G`): measured speedup `0.979×` (≈ no ordering benefit)

These runs confirm the ordering gain disappears when memory is increased enough to avoid eviction.

## Reproduce (tu)

```bash
ROOT="$HOME/bm_exp/bottlemod_cache_aware"
PY="$ROOT/.venv/bin/python"

# Base scale
PYTHONPATH="$ROOT" "$PY" \
  "$ROOT/thesis_experiment/02_memory_increase/exp2_memory_increase_no_eviction.py" \
  --video-a /mnt/sata/input_2g_a.mp4 \
  --video-b /mnt/sata/input_2g_b.mp4 \
  --mem-evict 3G --mem-no-evict 5G \
  --trials 3 --drop-caches \
  --out-dir "/var/tmp/exp1_noevict_base_$(date +%Y%m%d_%H%M%S)"

# 10× scale
PYTHONPATH="$ROOT" "$PY" \
  "$ROOT/thesis_experiment/02_memory_increase/exp2_memory_increase_no_eviction.py" \
  --video-a /mnt/sata/input_20g_a.mp4 \
  --video-b /mnt/sata/input_20g_b.mp4 \
  --mem-evict 30G --mem-no-evict 70G \
  --trials 2 --drop-caches \
  --out-dir "/var/tmp/exp1_noevict_10x_$(date +%Y%m%d_%H%M%S)"
```

## Findings

| Timestamp | Files | Memory | Notes |
|-----------|-------|--------|-------|
| `20260226_185746` | 1.9 GB × 2 | 3G / 5G | Base no-eviction validation |
| `20260226_200024` | 18.7 GB × 2 | 30G / 70G | 10× no-eviction validation |
