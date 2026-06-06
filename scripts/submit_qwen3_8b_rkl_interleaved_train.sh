#!/bin/bash
# Submit one RunAI job that trains a Qwen3-0.6B interleaved RKL draft for a
# Qwen3-8B target and saves checkpoints.

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/scratch/cs552-mnlp-kzy}"
REPO_DIR="${REPO_DIR:-${WORK_ROOT}/repos/distilled-draft-decoding-interleaved}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${WORK_ROOT}/checkpoints}"
HYDRA_OUTPUTS_DIR="${HYDRA_OUTPUTS_DIR:-${WORK_ROOT}/hydra}"
WANDB_DIR="${WANDB_DIR:-${WORK_ROOT}/wandb}"
DATA_DIR="${DATA_DIR:-/scratch/cs552-data}"

DATA="${DATA:-ultrachat_50k}"
LOSS_KIND="${LOSS_KIND:-rkl}"
TARGET_ID="${TARGET_ID:-Qwen/Qwen3-8B}"
DRAFT_ID="${DRAFT_ID:-Qwen/Qwen3-0.6B}"
ALPHA="${ALPHA:-1.0}"
TEMP="${TEMP:-1.0}"
SEED="${SEED:-42}"

STEPS="${STEPS:-500}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
LR="${LR:-1e-5}"
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

RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qwen3_8btarget_0p6b_interleaved_rkl}"
if [[ "${STEPS}" =~ ^[0-9]+$ ]] && [[ "${STEPS}" -gt 0 ]]; then
  STEP_TAG="s${STEPS}"
else
  STEP_TAG="e${EPOCHS}"
fi
RUN_NAME_PREFIX_TAGGED="${RUN_NAME_PREFIX}_${STEP_TAG}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${RUN_NAME_PREFIX_TAGGED}_${DATA}_seed${SEED}}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"
REPO_BRANCH="${REPO_BRANCH:-interleaved}"
REPO_URL="${REPO_URL:-https://github.com/roxanna-ke/distilled-draft-decoding.git}"

quote() {
  printf "%q" "$1"
}

run_command="DATA=$(quote "${DATA}")"
run_command+=" WORK_ROOT=$(quote "${WORK_ROOT}")"
run_command+=" CHECKPOINTS_ROOT=$(quote "${CHECKPOINTS_ROOT}")"
run_command+=" LOSS_KIND=$(quote "${LOSS_KIND}")"
run_command+=" TARGET_ID=$(quote "${TARGET_ID}")"
run_command+=" DRAFT_ID=$(quote "${DRAFT_ID}")"
run_command+=" ALPHA=$(quote "${ALPHA}")"
run_command+=" TEMP=$(quote "${TEMP}")"
run_command+=" SEED=$(quote "${SEED}")"
run_command+=" STEPS=$(quote "${STEPS}")"
run_command+=" EPOCHS=$(quote "${EPOCHS}")"
run_command+=" BATCH_SIZE=$(quote "${BATCH_SIZE}")"
run_command+=" GRAD_ACCUM_STEPS=$(quote "${GRAD_ACCUM_STEPS}")"
run_command+=" LR=$(quote "${LR}")"
run_command+=" MAX_SEQ_LEN=$(quote "${MAX_SEQ_LEN}")"
run_command+=" KD_CHUNK_SIZE=$(quote "${KD_CHUNK_SIZE}")"
run_command+=" COMPILE_TARGET=$(quote "${COMPILE_TARGET}")"
run_command+=" EVAL_REPORTING_STEPS=$(quote "${EVAL_REPORTING_STEPS}")"
run_command+=" PYTORCH_CUDA_ALLOC_CONF=$(quote "${PYTORCH_CUDA_ALLOC_CONF}")"
run_command+=" TRAIN_ROLLIN=$(quote "${TRAIN_ROLLIN}")"
run_command+=" INTERLEAVED_ROLLOUT_TOKENS=$(quote "${INTERLEAVED_ROLLOUT_TOKENS}")"
run_command+=" INTERLEAVED_STUDENT_MODE=$(quote "${INTERLEAVED_STUDENT_MODE}")"
run_command+=" INTERLEAVED_STUDENT_TEMPERATURE=$(quote "${INTERLEAVED_STUDENT_TEMPERATURE}")"
run_command+=" INTERLEAVED_STUDENT_TOP_P=$(quote "${INTERLEAVED_STUDENT_TOP_P}")"
run_command+=" INTERLEAVED_TEACHER_TOPK=$(quote "${INTERLEAVED_TEACHER_TOPK}")"
run_command+=" RUN_NAME_PREFIX=$(quote "${RUN_NAME_PREFIX}")"
run_command+=" EXPERIMENT_NAME=$(quote "${EXPERIMENT_NAME}")"
run_command+=" WANDB_GROUP=$(quote "${WANDB_GROUP}")"
run_command+=" bash scripts/run_qwen3_8b_rkl_interleaved_train.sh"

echo ">>> RUN_COMMAND chars: ${#run_command}"
echo ">>> Submitting Qwen3 8B/0.6B interleaved RKL train job: ${EXPERIMENT_NAME}"
REPO_URL="${REPO_URL}" \
REPO_BRANCH="${REPO_BRANCH}" \
REPO_DIR="${REPO_DIR}" \
CHECKPOINTS_DIR="${CHECKPOINTS_ROOT}" \
DATA_DIR="${DATA_DIR}" \
HYDRA_OUTPUTS_DIR="${HYDRA_OUTPUTS_DIR}" \
WANDB_DIR="${WANDB_DIR}" \
RUN_NAME="${EXPERIMENT_NAME}" \
RUN_COMMAND="${run_command}" \
./rcp_support/submit_train.sh "qwen3-8b-rkl-interleaved-train"
