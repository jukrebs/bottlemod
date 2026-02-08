#!/usr/bin/env bash
set -euo pipefail

# Re-plot from an existing fio run directory.
#
# Usage:
#   OUT_DIR=/var/tmp/bottlemod_exp_throttled_2g_cache4g bash scripts/plot_fio_results.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ROOT="$(repo_root)"
activate_venv_if_present

OUT_DIR="${OUT_DIR:-}"
if [[ -z "$OUT_DIR" ]]; then
  echo "ERROR: OUT_DIR must be set" >&2
  exit 1
fi

DRAM_CAPACITY="${DRAM_CAPACITY:-}"

ARGS=("--in-dir" "$OUT_DIR" "--out" "$ROOT/experiment_ground_truth_fio.json")
if [[ -n "$DRAM_CAPACITY" ]]; then
  ARGS+=("--dram-capacity" "$DRAM_CAPACITY")
fi

python "$ROOT/fio_parse_and_model.py" "${ARGS[@]}"
python "$ROOT/experiment_plots.py"

echo "Done. Plots: $ROOT/experiment_plots/*.png"
