# Experiment 01 — Cache-Aware Task Reordering

This experiment evaluates task ordering effects under constrained page cache using:

- Script: `exp1_reordering.py`
- Workflow: `A-B-A-B` (interleaved) vs `A-A-B-B` (grouped)
- Prediction models: measured ffmpeg, vanilla BottleMod, BottleMod-CA

## Reproduce (tu)

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

Legacy sweep results produced with removed `exp1_cache_aware_ordering.py` are kept only as historical findings under `findings/`.
