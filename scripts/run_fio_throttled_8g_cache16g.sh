#!/usr/bin/env bash
set -euo pipefail

# Larger run for stronger signal (more time, more IO). Use only if you have enough free RAM.
#
# Defaults:
# - File size 8G
# - Cache budget 16G (MemoryMax)
# - Buffered I/O, systemd-run IO throttling
#
# Usage:
#   bash scripts/run_fio_throttled_8g_cache16g.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ROOT="$(repo_root)"
activate_venv_if_present

DEVICE="${DEVICE:-/dev/nvme0n1}"
OUT_DIR="${OUT_DIR:-/var/tmp/bottlemod_exp_throttled_8g_cache16g}"
TRIALS="${TRIALS:-5}"
SIZE="${SIZE:-8G}"

MEMORY_MAX="${MEMORY_MAX:-16G}"
CPU_AFFINITY="${CPU_AFFINITY:-0-3}"

RBPS="${RBPS:-300M}"

echo "Throttled large: DEVICE=$DEVICE OUT_DIR=$OUT_DIR SIZE=$SIZE TRIALS=$TRIALS MemoryMax=$MEMORY_MAX"
echo "Set FORCE=1 to skip confirmation."
confirm_or_exit "Proceed with throttled large fio run on OS disk?"

python "$ROOT/fio_runner.py" \
  --out-dir "$OUT_DIR" \
  --size "$SIZE" \
  --trials "$TRIALS" \
  --systemd-run --systemd-sudo \
  --systemd-property "CPUAffinity=$CPU_AFFINITY" \
  --systemd-property "MemoryMax=$MEMORY_MAX" \
  --systemd-property "MemorySwapMax=0" \
  --systemd-property "IOReadBandwidthMax=$DEVICE $RBPS" \
  --rand-numjobs 0

python "$ROOT/fio_parse_and_model.py" \
  --in-dir "$OUT_DIR" \
  --dram-capacity "$MEMORY_MAX" \
  --out "$ROOT/experiment_ground_truth_fio.json"

python "$ROOT/experiment_plots.py"

echo "Done. Plots: $ROOT/experiment_plots/*.png"
