#!/usr/bin/env bash
set -euo pipefail

# Quick smoke run (safe defaults for OS disk):
# - buffered I/O (--direct=0)
# - cgroup v2 limits via systemd-run (system scope; requires sudo)
# - small working set to make warm cache meaningful
#
# Usage:
#   bash scripts/run_fio_throttled_smoke.sh
#
# Overrides (env):
#   DEVICE=/dev/nvme0n1 OUT_DIR=/var/tmp/bottlemod_exp_smoke TRIALS=1 SIZE=2G
#   MEMORY_MAX=4G CPU_AFFINITY=0-3 RBPS=200M

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ROOT="$(repo_root)"
activate_venv_if_present

require_file "$ROOT/fio_runner.py"
require_file "$ROOT/fio_parse_and_model.py"
require_file "$ROOT/experiment_plots.py"

DEVICE="${DEVICE:-/dev/nvme0n1}"
OUT_DIR="${OUT_DIR:-/var/tmp/bottlemod_exp_smoke}"
TRIALS="${TRIALS:-1}"
SIZE="${SIZE:-2G}"

MEMORY_MAX="${MEMORY_MAX:-4G}"
CPU_AFFINITY="${CPU_AFFINITY:-0-3}"

RBPS="${RBPS:-200M}"

echo "Running throttled fio smoke on DEVICE=$DEVICE (OS disk)."
echo "Set FORCE=1 to skip confirmation."
confirm_or_exit "Proceed with throttled fio run?"

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
