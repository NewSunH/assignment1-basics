#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root even if called from another directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p slurm_logs outputs

run_uv() {
  if command -v uv >/dev/null 2>&1; then
    uv run "$@"
    return
  fi

  # Fallback: try the local venv (if you created one on the cluster)
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    "$REPO_ROOT/.venv/bin/python" "$@"
    return
  fi

  echo "ERROR: 'uv' not found and .venv missing. Install uv or create a venv." >&2
  echo "Hint: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
}

print_env() {
  echo "HOST=$(hostname)"
  echo "PWD=$PWD"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"
  nvidia-smi || true
}
