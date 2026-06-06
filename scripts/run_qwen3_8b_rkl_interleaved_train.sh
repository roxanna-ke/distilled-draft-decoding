#!/bin/bash
# Run Qwen3-8B / Qwen3-0.6B interleaved RKL training and save checkpoints.
# This script is intended to run inside the RunAI pod from the checked-out repo.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ENV_SH="${KDSD_ENV_SH:-${ROOT}/scripts/env.sh}"
source "${ENV_SH}"
echo ">>> Sourced env: ${ENV_SH}"
echo ">>> Python: ${KDSD_PYTHON}"

WORK_ROOT="${WORK_ROOT:-/scratch/cs552-mnlp-kzy}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${CHECKPOINTS_DIR:-${WORK_ROOT}/checkpoints}}"
RESULTS_DIR_ROOT="${RESULTS_DIR_ROOT:-${WORK_ROOT}/results}"

DATA="${DATA:-ultrachat_50k}"
LOSS_KIND="${LOSS_KIND:-rkl}"
TARGET_ID="${TARGET_ID:-Qwen/Qwen3-8B}"
DRAFT_ID="${DRAFT_ID:-Qwen/Qwen3-0.6B}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qwen3_8btarget_0p6b_interleaved_rkl}"
SEED="${SEED:-42}"

STEPS="${STEPS:-500}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
LR="${LR:-1e-5}"
ALPHA="${ALPHA:-1.0}"
TEMP="${TEMP:-1.0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
KD_CHUNK_SIZE="${KD_CHUNK_SIZE:-128}"
COMPILE_TARGET="${COMPILE_TARGET:-false}"
EVAL_REPORTING_STEPS="${EVAL_REPORTING_STEPS:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TRAIN_ROLLIN="${TRAIN_ROLLIN:-interleaved}"
INTERLEAVED_ROLLOUT_TOKENS="${INTERLEAVED_ROLLOUT_TOKENS:-32}"
INTERLEAVED_STUDENT_MODE="${INTERLEAVED_STUDENT_MODE:-greedy}"
INTERLEAVED_STUDENT_TEMPERATURE="${INTERLEAVED_STUDENT_TEMPERATURE:-0.3}"
INTERLEAVED_STUDENT_TOP_P="${INTERLEAVED_STUDENT_TOP_P:-1.0}"
INTERLEAVED_TEACHER_TOPK="${INTERLEAVED_TEACHER_TOPK:-1}"

if [[ "${STEPS}" =~ ^[0-9]+$ ]] && [[ "${STEPS}" -gt 0 ]]; then
  STEP_TAG="s${STEPS}"
else
  STEP_TAG="e${EPOCHS}"
fi
RUN_NAME_PREFIX_TAGGED="${RUN_NAME_PREFIX}_${STEP_TAG}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${RUN_NAME_PREFIX_TAGGED}_${DATA}_seed${SEED}}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"

echo ">>> Experiment: ${EXPERIMENT_NAME}"
echo ">>> Work root: ${WORK_ROOT}"
echo ">>> Checkpoints root: ${CHECKPOINTS_ROOT}"
echo ">>> Eval results root: ${RESULTS_DIR_ROOT}"
echo ">>> Target: ${TARGET_ID}"
echo ">>> Draft init: ${DRAFT_ID}"
echo ">>> Data: ${DATA}"
echo ">>> Loss: ${LOSS_KIND}"
echo ">>> alpha/temp: ${ALPHA}/${TEMP}"
echo ">>> train steps/epochs: ${STEPS}/${EPOCHS}"
echo ">>> train batch/grad accum: ${BATCH_SIZE}/${GRAD_ACCUM_STEPS}"
echo ">>> train rollin: ${TRAIN_ROLLIN}"
echo ">>> interleaved rollout/top-k: ${INTERLEAVED_ROLLOUT_TOKENS}/${INTERLEAVED_TEACHER_TOPK}"

export PYTORCH_CUDA_ALLOC_CONF
export WORK_ROOT CHECKPOINTS_ROOT
mkdir -p "${CHECKPOINTS_ROOT}"

run_name="${RUN_NAME_PREFIX_TAGGED}_${DATA}_a${ALPHA}_seed${SEED}"
checkpoint_dir="${CHECKPOINTS_ROOT}/${run_name}"
echo ">>> Starting training: ${run_name}"

WANDB_GROUP="${WANDB_GROUP}" \
WANDB_NAME="${run_name}" \
WANDB_JOB_TYPE="train" \
"${KDSD_PYTHON}" scripts/train.py \
  model=qwen3 train=a100_40gb_qwen3 "data=${DATA}" "loss=${LOSS_KIND}" \
  "model.target=${TARGET_ID}" \
  "model.draft_default=${DRAFT_ID}" \
  "train.draft_init=${DRAFT_ID}" \
  "train.max_steps=${STEPS}" \
  "train.num_train_epochs=${EPOCHS}" \
  "train.per_device_train_batch_size=${BATCH_SIZE}" \
  "train.per_device_eval_batch_size=${BATCH_SIZE}" \
  "train.gradient_accumulation_steps=${GRAD_ACCUM_STEPS}" \
  "train.learning_rate=${LR}" \
  "train.compile_target=${COMPILE_TARGET}" \
  "train.eval_steps=${EVAL_REPORTING_STEPS}" \
  "train.rollin=${TRAIN_ROLLIN}" \
  "train.interleaved_rollout_tokens=${INTERLEAVED_ROLLOUT_TOKENS}" \
  "train.interleaved_student_mode=${INTERLEAVED_STUDENT_MODE}" \
  "train.interleaved_student_temperature=${INTERLEAVED_STUDENT_TEMPERATURE}" \
  "train.interleaved_student_top_p=${INTERLEAVED_STUDENT_TOP_P}" \
  "train.interleaved_teacher_topk=${INTERLEAVED_TEACHER_TOPK}" \
  "data.max_seq_len=${MAX_SEQ_LEN}" \
  "loss.alpha=${ALPHA}" \
  "loss.temperature=${TEMP}" \
  "loss.chunk_size=${KD_CHUNK_SIZE}" \
  "seed=${SEED}" \
  "output_dir=${checkpoint_dir}" \
  "run_name=${run_name}"

echo ">>> Finished training: ${run_name}"
echo ">>> Checkpoint saved under: ${checkpoint_dir}"
