#!/usr/bin/env bash
#SBATCH --job-name=ts_ln_ablate
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

# Adjust these for your cluster
#SBATCH --partition=lfs-dev-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# One LR per task; cap concurrency if needed
#SBATCH --array=0-5%8

set -euo pipefail

source scripts/slurm/_common.sh
print_env

TRAIN_TOKENS="outputs/tinystories_train_tokens.uint16"
VALID_TOKENS="outputs/tinystories_valid_tokens.uint16"

# Include the "previous optimal" LR=3e-3 plus smaller LRs to regain stability.
LRS=(3e-3 1e-3 5e-4 3e-4 1e-4 3e-5)
LR="${LRS[$SLURM_ARRAY_TASK_ID]}"

SEED="${SEED:-0}"
DTYPE="${DTYPE:-bf16}"

# Use a shorter run for the sweep to quickly find a stable LR.
STEPS="${STEPS:-2000}"
WARMUP="${WARMUP:-200}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200}"
EVAL_BATCHES="${EVAL_BATCHES:-200}"

# For divergence curves, keep early logging dense.
LOG_INTERVAL="${LOG_INTERVAL:-1}"

NAME_PREFIX="layer_norm_ablation"

echo "[layer_norm_ablation] use_rmsnorm=false lr=$LR steps=$STEPS seed=$SEED dtype=$DTYPE"

run_uv python -u experiments/train_tinystories_lm.py \
  --train-tokens "$TRAIN_TOKENS" \
  --valid-tokens "$VALID_TOKENS" \
  --no-rmsnorm \
  --lr "$LR" \
  --steps "$STEPS" \
  --warmup-steps "$WARMUP" \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-batches "$EVAL_BATCHES" \
  --log-interval "$LOG_INTERVAL" \
  --seed "$SEED" \
  --dtype "$DTYPE" \
  --run-name "${NAME_PREFIX}_norn_lr${LR}_steps${STEPS}_seed${SEED}"

echo "Done lr=$LR"
