#!/usr/bin/env bash
#SBATCH --job-name=ts_pretok
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Adjust these for your cluster
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# No GPU needed for pretokenization; if your cluster requires CPU partition, change above.

set -euo pipefail
source scripts/slurm/_common.sh

print_env

# Assumes you already trained/placed a TinyStories tokenizer at outputs/tinystories_{vocab,merges}.*
# If you use a different tokenizer path, edit these.
VOCAB="outputs/tinystories_vocab.json"
MERGES="outputs/tinystories_merges.txt"

run_uv experiments/pretokenize_dataset.py \
  --vocab "$VOCAB" \
  --merges "$MERGES" \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output outputs/tinystories_train_tokens.uint16 \
  --add-eos

run_uv experiments/pretokenize_dataset.py \
  --vocab "$VOCAB" \
  --merges "$MERGES" \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output outputs/tinystories_valid_tokens.uint16 \
  --add-eos

echo "Done pretokenization."
ls -lh outputs/tinystories_*tokens.* || true
