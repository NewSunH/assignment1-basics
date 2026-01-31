#!/usr/bin/env bash
#SBATCH --job-name=ts_bs
#SBATCH --partition=lfs-dev-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/ts_bs_%j.out
#SBATCH --error=slurm_logs/ts_bs_%j.err
#SBATCH --array=0-6

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Batch sizes to try (adjust the upper end as you discover the GPU memory limit).
BATCH_SIZES=(1 4 16 64 128 256 512)

BS=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}
CTX=256

# Heuristic: keep a roughly fixed token budget per run so bs=1 is feasible.
# steps = clamp(tokens_target / (bs * ctx), min_steps, max_steps)
TOKENS_TARGET=20000000
MIN_STEPS=200
MAX_STEPS=20000

# Start from the best lr found at batch_size=128, then apply linear scaling.
BASE_BS=128
BASE_LR=3e-3

export BS CTX TOKENS_TARGET MIN_STEPS MAX_STEPS BASE_BS BASE_LR
read -r STEPS LR_MAX < <(uv run python - <<'PY'
import os

bs = int(os.environ["BS"])
ctx = int(os.environ["CTX"])
tokens_target = int(os.environ["TOKENS_TARGET"])
min_steps = int(os.environ["MIN_STEPS"])
max_steps = int(os.environ["MAX_STEPS"])
base_bs = int(os.environ["BASE_BS"])
base_lr = float(os.environ["BASE_LR"])

steps = int(tokens_target // (bs * ctx))
steps = max(min_steps, min(max_steps, steps))

# linear scaling rule
lr = base_lr * (bs / base_bs)

# avoid the clearly-bad regime seen at lr_max≈3e-2 for this setup
lr = min(lr, 2e-2)

print(steps, lr)
PY
)

# Pick intervals so you get multiple eval points regardless of step count.
export STEPS
EVAL_INTERVAL=$(uv run python - <<'PY'
import os

steps = int(os.environ["STEPS"])
print(max(10, steps // 20))
PY
)
LOG_INTERVAL=$(uv run python - <<'PY'
import os

steps = int(os.environ["STEPS"])
print(max(1, steps // 200))
PY
)

echo "[batch_size_experiment] bs=${BS} steps=${STEPS} lr_max=${LR_MAX} eval_interval=${EVAL_INTERVAL} log_interval=${LOG_INTERVAL}"

uv run python experiments/train_tinystories_lm.py \
  --train-tokens outputs/tinystories_train_tokens.uint16 \
  --valid-tokens outputs/tinystories_valid_tokens.uint16 \
  --context-length ${CTX} \
  --batch-size ${BS} \
  --steps ${STEPS} \
  --lr ${LR_MAX} \
  --warmup-steps 200 \
  --eval-interval ${EVAL_INTERVAL} \
  --log-interval ${LOG_INTERVAL} \
  --eval-batches 200 \
  --dtype bf16 \
  --seed 0
