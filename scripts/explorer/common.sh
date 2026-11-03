#!/usr/bin/env bash
# Shared Explorer env. Prefer an existing venv; do not create conda.
# Override with: export PHIL_VENV=/path/to/venv
set -euo pipefail

PHIL_SCRATCH="${PHIL_SCRATCH:-/scratch/galoaa.b/phillm}"
PHIL_CODE="${PHIL_CODE:-$PHIL_SCRATCH/Phil}"
export HF_HOME="${HF_HOME:-$PHIL_SCRATCH/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTHONPATH="$PHIL_CODE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_HOME" "$PHIL_SCRATCH"/{logs,checkpoints,preds,probes,weightspace}

discover_venv() {
  if [[ -n "${PHIL_VENV:-}" && -f "${PHIL_VENV}/bin/activate" ]]; then
    echo "$PHIL_VENV"
    return
  fi
  local candidates=(
    /scratch/galoaa.b/egoexo360/env
    /scratch/galoaa.b/egoexo360/env_agent
    "$PHIL_SCRATCH/venv"
    /scratch/galoaa.b/venv
    /scratch/galoaa.b/hf/venv
    /home/galoaa.b/venv
    /home/galoaa.b/.venv
  )
  # also any scratch project venv
  local d
  for d in /scratch/galoaa.b/*/venv /scratch/galoaa.b/*/.venv; do
    candidates+=("$d")
  done
  for d in "${candidates[@]}"; do
    if [[ -f "$d/bin/activate" ]]; then
      echo "$d"
      return
    fi
  done
  echo ""
}

activate_phil_venv() {
  local v
  v="$(discover_venv)"
  if [[ -z "$v" ]]; then
    echo "No existing venv found. Creating $PHIL_SCRATCH/venv with python3 -m venv" >&2
    python3 -m venv "$PHIL_SCRATCH/venv"
    v="$PHIL_SCRATCH/venv"
  fi
  # shellcheck disable=SC1090
  source "$v/bin/activate"
  echo "Using venv: $v"
  python -m pip install -q -U pip
  python -m pip install -q -e "$PHIL_CODE[train,dev]" || python -m pip install -q -e "$PHIL_CODE"
}
