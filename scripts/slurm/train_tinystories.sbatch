#!/usr/bin/env bash
#SBATCH --job-name=ts_train
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Adjust these for your cluster
#SBATCH --partition=lfs-dev-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/assignment1-basics}"

TRAIN_TOKENS="outputs/tinystories_train_tokens.uint16"
VALID_TOKENS="outputs/tinystories_valid_tokens.uint16"

# Override by exporting LR=... before sbatch, e.g.:
#   LR=3e-4 sbatch scripts/slurm/train_tinystories.sbatch
LR="${LR:-3e-4}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-bf16}"

echo "Starting run: lr=$LR seed=$SEED dtype=$DTYPE"

# Capture the first line summary dict printed by the script so we can extract the run_dir.
SUMMARY_LINE="$(
  uv run python -u experiments/train_tinystories_lm.py \
    --train-tokens "$TRAIN_TOKENS" \
    --valid-tokens "$VALID_TOKENS" \
    --lr "$LR" \
    --seed "$SEED" \
    --dtype "$DTYPE" \
  | head -n 1
)"

echo "$SUMMARY_LINE"

RUN_DIR="$(echo "$SUMMARY_LINE" | sed -n "s/.*'run_dir': '\([^']\+\)'.*/\1/p")"
if [[ -n "$RUN_DIR" ]]; then
  echo "This run_dir: $RUN_DIR"
  echo "This lr: $LR"
  echo "Artifacts: $RUN_DIR/{run.json,metrics.jsonl,summary.json,checkpoints/}"
else
  echo "WARNING: could not parse run_dir from summary line." >&2
  echo "Full line: $SUMMARY_LINE" >&2
fi

exit 0
  --train-tokens "$TRAIN_TOKENS" \
  --valid-tokens "$VALID_TOKENS" \
  --lr "$LR" \
  --seed "$SEED" \
  --dtype "$DTYPE"

echo "Run finished. Latest runs:"
ls -1dt outputs/runs/* | head -n 5 || true
