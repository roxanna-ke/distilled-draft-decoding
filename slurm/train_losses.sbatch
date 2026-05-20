#!/bin/bash
# Train CE/FKL/RKL/JSD draft-model variants on Izar.
#
# Submit:
#   sbatch slurm/train_losses.sbatch
#
# Optional overrides at submit time:
#   TRAIN_STEPS=4000 MAX_LENGTH=512 KD_TEMPERATURE=1.0 sbatch slurm/train_losses.sbatch
#   WANDB_PROJECT=my-project sbatch slurm/train_losses.sbatch

#SBATCH --job-name=kdsd-train-losses
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export SCRATCH_DIR="/scratch/izar/ke/nlp"
export HF_HOME="$SCRATCH_DIR/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export WANDB_DIR="$SCRATCH_DIR/wandb"
export UV_CACHE_DIR="$SCRATCH_DIR/uv_cache"
export UV_LINK_MODE=copy

export WANDB_PROJECT="${WANDB_PROJECT:-kdsd}"
export WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p \
  "$SCRATCH_DIR" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$WANDB_DIR" \
  "$UV_CACHE_DIR" \
  "$SCRATCH_DIR/checkpoints" \
  logs

LOSSES=(ce fkl rkl jsd)
LOSS="${LOSSES[$SLURM_ARRAY_TASK_ID]}"

DATASET="${DATASET:-ultrachat_10k}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
MAX_LENGTH="${MAX_LENGTH:-512}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
KD_TEMPERATURE="${KD_TEMPERATURE:-4.0}"
KD_ALPHA="${KD_ALPHA:-0.5}"
SEED="${SEED:-42}"

RUN_NAME="kd_${LOSS}_${DATASET}_t${KD_TEMPERATURE}_seed${SEED}"
OUT_DIR="$SCRATCH_DIR/checkpoints/$RUN_NAME"

echo "repo=$SLURM_SUBMIT_DIR"
echo "run_name=$RUN_NAME"
echo "loss=$LOSS"
echo "dataset=$DATASET"
echo "output=$OUT_DIR"
echo "wandb_project=$WANDB_PROJECT"
echo "wandb_mode=$WANDB_MODE"
echo "temperature=$KD_TEMPERATURE alpha=$KD_ALPHA"


uv run python scripts/train.py \
  data="$DATASET" \
  loss="$LOSS" \
  seed="$SEED" \
  run_name="$RUN_NAME" \
  model.dtype=float16 \
  model.device=cuda \
  loss.temperature="$KD_TEMPERATURE" \
  loss.alpha="$KD_ALPHA" \
  train.output_dir="$OUT_DIR" \
  train.steps="$TRAIN_STEPS" \
  train.max_length="$MAX_LENGTH" \
  train.batch_size="$BATCH_SIZE" \
  train.gradient_accumulation_steps="$GRAD_ACCUM" \
  train.learning_rate="$LEARNING_RATE" \
  train.eval_steps=100 \
  train.save_steps=500 \
  train.max_eval_batches=16 \
  train.wandb.enabled=true \
  train.wandb.project="$WANDB_PROJECT" \
  train.wandb.mode="$WANDB_MODE"
