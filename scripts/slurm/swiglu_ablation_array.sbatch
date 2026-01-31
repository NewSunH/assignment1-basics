#!/usr/bin/env bash
#SBATCH --job-name=ts_swiglu_ablate
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

# Adjust these for your cluster
#SBATCH --partition=lfs-dev-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Grid: 2 variants x 5 LRs = 10 tasks
#SBATCH --array=0-9%10

set -euo pipefail

source scripts/slurm/_common.sh
print_env

TRAIN_TOKENS="outputs/tinystories_train_tokens.uint16"
VALID_TOKENS="outputs/tinystories_valid_tokens.uint16"

VARIANTS=(swiglu silu)
LRS=(3e-3 1e-3 5e-4 3e-4 1e-4)

variant_idx=$((SLURM_ARRAY_TASK_ID / ${#LRS[@]}))
lr_idx=$((SLURM_ARRAY_TASK_ID % ${#LRS[@]}))

FFN_VARIANT="${VARIANTS[$variant_idx]}"
LR="${LRS[$lr_idx]}"

SEED="${SEED:-0}"
DTYPE="${DTYPE:-bf16}"

STEPS="${STEPS:-2000}"
WARMUP="${WARMUP:-200}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200}"
EVAL_BATCHES="${EVAL_BATCHES:-200}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

NAME_PREFIX="swiglu_ablation"

echo "[swiglu_ablation] ffn_variant=$FFN_VARIANT lr=$LR steps=$STEPS seed=$SEED dtype=$DTYPE"

run_uv python -u experiments/train_tinystories_lm.py \
  --train-tokens "$TRAIN_TOKENS" \
  --valid-tokens "$VALID_TOKENS" \
  --ffn-variant "$FFN_VARIANT" \
  --lr "$LR" \
  --steps "$STEPS" \
  --warmup-steps "$WARMUP" \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-batches "$EVAL_BATCHES" \
  --log-interval "$LOG_INTERVAL" \
  --seed "$SEED" \
  --dtype "$DTYPE" \
  --run-name "${NAME_PREFIX}_${FFN_VARIANT}_lr${LR}_steps${STEPS}_seed${SEED}"

echo "Done ffn_variant=$FFN_VARIANT lr=$LR"
