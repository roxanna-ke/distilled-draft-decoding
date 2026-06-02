#!/usr/bin/env bash
# Run the same non-CE KD loss ablation as run_loss_size_ablation.sh, but train and
# validate on the UltraChat 10k split. Final SD eval defaults to the 50k held-out
# eval split so 10k and 50k trained models are directly comparable.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Train/val come from configs/data/ultrachat_10k.yaml:
#   train_path = /scratch/cs552-data/processed/ultrachat_10k/train.jsonl
#   val_path   = /scratch/cs552-data/processed/ultrachat_10k/val.jsonl
export DATA="${DATA:-ultrachat_10k}"

# Final SD eval defaults to the 50k config's held-out eval split:
#   /scratch/cs552-data/processed/ultrachat_50k/eval.jsonl
export EVAL_DATA="${EVAL_DATA:-ultrachat_50k}"

train_steps="${TRAIN_STEPS:-8000}"
seq_len="${DATA_MAX_SEQ_LEN:-512}"
train_bs="${TRAIN_BATCH_SIZE:-2}"
grad_accum="${GRAD_ACCUM_STEPS:-4}"
export RUN_TAG="${RUN_TAG:-ultra10k_bugfix_s${train_steps}_seq${seq_len}_effbs$((train_bs * grad_accum))}"

exec bash "${ROOT}/scripts/run_loss_size_ablation.sh"
