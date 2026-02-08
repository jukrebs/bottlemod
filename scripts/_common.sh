#!/usr/bin/env bash
set -euo pipefail

repo_root() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  (cd "$here/.." && pwd)
}

activate_venv_if_present() {
  local root
  root="$(repo_root)"

  if [[ -f "$root/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$root/.venv/bin/activate"
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required file: $path" >&2
    exit 1
  fi
}

confirm_or_exit() {
  local msg="$1"
  if [[ "${FORCE:-}" == "1" ]]; then
    return
  fi
  read -r -p "$msg [y/N] " ans
  case "$ans" in
    y|Y|yes|YES) ;;
    *) echo "Aborted." >&2; exit 1 ;;
  esac
}
