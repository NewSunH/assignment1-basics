#!/usr/bin/env bash
#SBATCH --job-name=ts_train
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Adjust these for your cluster
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
source scripts/slurm/_common.sh

print_env

TRAIN_TOKENS="outputs/tinystories_train_tokens.uint16"
VALID_TOKENS="outputs/tinystories_valid_tokens.uint16"

# Override by exporting LR=... before sbatch, e.g.:
#   LR=3e-4 sbatch scripts/slurm/train_tinystories.sbatch
LR="${LR:-3e-4}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-bf16}"

run_uv experiments/train_tinystories_lm.py \
  --train-tokens "$TRAIN_TOKENS" \
  --valid-tokens "$VALID_TOKENS" \
  --lr "$LR" \
  --seed "$SEED" \
  --dtype "$DTYPE" \
  --compile

echo "Run finished. Latest runs:"
ls -1dt outputs/runs/* | head -n 5 || true
