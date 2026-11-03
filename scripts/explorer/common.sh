#!/usr/bin/env bash
# Shared Explorer env. Prefer existing venv (no conda).
set -euo pipefail

PHIL_SCRATCH="${PHIL_SCRATCH:-/scratch/galoaa.b/phillm}"
PHIL_CODE="${PHIL_CODE:-$PHIL_SCRATCH/Phil}"
export HF_HOME="${HF_HOME:-$PHIL_SCRATCH/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTHONPATH="$PHIL_CODE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_HOME" "$PHIL_SCRATCH"/{logs,checkpoints,preds,probes,weightspace}

DEFAULT_VENV="/scratch/galoaa.b/phillm/venv"
FALLBACK_VENV="/scratch/galoaa.b/egoexo360/env"

activate_phil_venv() {
  local v="${PHIL_VENV:-$DEFAULT_VENV}"
  if [[ ! -x "$v/bin/python" ]]; then
    v="$FALLBACK_VENV"
  fi
  if [[ ! -x "$v/bin/python" ]]; then
    echo "ERROR: python not found in $v; run scripts/explorer/bootstrap_venv.sh" >&2
    exit 1
  fi
  if [[ -f "$v/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$v/bin/activate"
  else
    export PATH="$v/bin:$PATH"
    export VIRTUAL_ENV="$v"
  fi
  hash -r
  echo "Using venv: $v"
  echo "python=$(command -v python) version=$(python -V 2>&1)"
  export PYTHONPATH="$PHIL_CODE${PYTHONPATH:+:$PYTHONPATH}"
  # Avoid pip editable installs on broken site-packages; rely on PYTHONPATH.
}
