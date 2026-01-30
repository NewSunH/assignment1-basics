#!/usr/bin/env bash
#SBATCH --job-name=ts_lr_sweep
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

# Adjust these for your cluster
#SBATCH --partition=lfs-dev-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Array: one LR per task. Limit concurrency to 8 for an 8-GPU node.
#SBATCH --array=0-5%8

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/assignment1-basics}"

TRAIN_TOKENS="outputs/tinystories_train_tokens.uint16"
VALID_TOKENS="outputs/tinystories_valid_tokens.uint16"

# Edit this LR list to include at least one divergent run.
LRS=(1e-4 2e-4 3e-4 5e-4 8e-4 1e-3)
LR="${LRS[$SLURM_ARRAY_TASK_ID]}"

SEED="${SEED:-0}"
DTYPE="${DTYPE:-bf16}"

NAME_PREFIX="learning_rate_array"

uv run python experiments/train_tinystories_lm.py \
  --train-tokens "$TRAIN_TOKENS" \
  --valid-tokens "$VALID_TOKENS" \
  --lr "$LR" \
  --seed "$SEED" \
  --dtype "$DTYPE" \
  --run-name "${NAME_PREFIX}_lr${LR}_seed${SEED}"

echo "Done lr=$LR"
