#!/usr/bin/env bash
#SBATCH --job-name=ts_ln_ablate_best
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Adjust these for your cluster
#SBATCH --partition=lfs-dev-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

source scripts/slurm/_common.sh
print_env

TRAIN_TOKENS="outputs/tinystories_train_tokens.uint16"
VALID_TOKENS="outputs/tinystories_valid_tokens.uint16"

# Override before sbatch, e.g.:
#   LR=3e-4 STEPS=10000 sbatch scripts/slurm/layer_norm_ablation_best.sbatch
LR="${LR:-3e-4}"
STEPS="${STEPS:-10000}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-bf16}"

WARMUP="${WARMUP:-200}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200}"
EVAL_BATCHES="${EVAL_BATCHES:-200}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

NAME_PREFIX="layer_norm_ablation"

echo "[layer_norm_ablation BEST] use_rmsnorm=false lr=$LR steps=$STEPS seed=$SEED dtype=$DTYPE"

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
  --run-name "${NAME_PREFIX}_BEST_norn_lr${LR}_steps${STEPS}_seed${SEED}"
