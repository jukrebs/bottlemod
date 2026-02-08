#!/usr/bin/env bash
set -euo pipefail

# Reproducible throttled baseline (recommended default on OS disk).
#
# Characteristics:
# - File size 2G, cache budget 4G -> warm pass should be meaningfully cache-resident
# - Buffered I/O (--direct=0)
# - systemd-run system scope IO throttling (sudo)
#
# Usage:
#   bash scripts/run_fio_throttled_2g_cache4g.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ROOT="$(repo_root)"
activate_venv_if_present

DEVICE="${DEVICE:-/dev/nvme0n1}"
OUT_DIR="${OUT_DIR:-/var/tmp/bottlemod_exp_throttled_2g_cache4g}"
TRIALS="${TRIALS:-5}"
SIZE="${SIZE:-2G}"

MEMORY_MAX="${MEMORY_MAX:-4G}"
CPU_AFFINITY="${CPU_AFFINITY:-0-3}"

RBPS="${RBPS:-200M}"

echo "Throttled baseline: DEVICE=$DEVICE OUT_DIR=$OUT_DIR SIZE=$SIZE TRIALS=$TRIALS MemoryMax=$MEMORY_MAX"
echo "Set FORCE=1 to skip confirmation."
confirm_or_exit "Proceed with throttled fio baseline?"

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
