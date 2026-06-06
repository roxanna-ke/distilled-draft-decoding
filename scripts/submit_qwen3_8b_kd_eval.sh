#!/bin/bash
# Submit one RunAI job that sequentially trains fkl/rkl/jsd drafts on UltraChat
# 50k with alpha=1, then evaluates them with vLLM speculative decoding.

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/scratch/cs552-mnlp-kzy}"
REPO_DIR="${REPO_DIR:-${WORK_ROOT}/repos/distilled-draft-decoding-interleaved}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${WORK_ROOT}/checkpoints}"
RESULTS_DIR_ROOT="${RESULTS_DIR_ROOT:-${WORK_ROOT}/results}"
HYDRA_OUTPUTS_DIR="${HYDRA_OUTPUTS_DIR:-${WORK_ROOT}/hydra}"
WANDB_DIR="${WANDB_DIR:-${WORK_ROOT}/wandb}"
DATA_DIR="${DATA_DIR:-/scratch/cs552-data}"

DATA="${DATA:-ultrachat_50k}"
LOSSES="${LOSSES:-fkl rkl jsd}"
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

RUN_EVAL="${RUN_EVAL:-true}"
EVAL_PRETRAINED_BASELINE="${EVAL_PRETRAINED_BASELINE:-true}"
EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-/scratch/cs552-data/processed/${DATA}/eval.jsonl}"
EVAL_PROMPTS_LIMIT="${EVAL_PROMPTS_LIMIT:-256}"
EVAL_GAMMA="${EVAL_GAMMA:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_MODE="${EVAL_MODE:-sampling}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.9}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-3}"
EVAL_REQUEST_BATCH_SIZE="${EVAL_REQUEST_BATCH_SIZE:-8}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-2048}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.9}"
EVAL_REPORT_TO_WANDB="${EVAL_REPORT_TO_WANDB:-true}"

TRAIN_ROLLIN="${TRAIN_ROLLIN:-interleaved}"
INTERLEAVED_ROLLOUT_TOKENS="${INTERLEAVED_ROLLOUT_TOKENS:-32}"
INTERLEAVED_STUDENT_MODE="${INTERLEAVED_STUDENT_MODE:-greedy}"
INTERLEAVED_STUDENT_TEMPERATURE="${INTERLEAVED_STUDENT_TEMPERATURE:-0.3}"
INTERLEAVED_STUDENT_TOP_P="${INTERLEAVED_STUDENT_TOP_P:-1.0}"
INTERLEAVED_TEACHER_TOPK="${INTERLEAVED_TEACHER_TOPK:-1}"

RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qwen3_8btarget_0p6b_interleaved}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${RUN_NAME_PREFIX}_${DATA}_seed${SEED}}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"
REPO_BRANCH="${REPO_BRANCH:-interleaved}"
REPO_URL="${REPO_URL:-https://github.com/roxanna-ke/distilled-draft-decoding.git}"

quote() {
  printf "%q" "$1"
}

run_command="DATA=$(quote "${DATA}")"
run_command+=" WORK_ROOT=$(quote "${WORK_ROOT}")"
run_command+=" CHECKPOINTS_ROOT=$(quote "${CHECKPOINTS_ROOT}")"
run_command+=" RESULTS_DIR_ROOT=$(quote "${RESULTS_DIR_ROOT}")"
run_command+=" LOSSES=$(quote "${LOSSES}")"
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
run_command+=" RUN_EVAL=$(quote "${RUN_EVAL}")"
run_command+=" EVAL_PRETRAINED_BASELINE=$(quote "${EVAL_PRETRAINED_BASELINE}")"
run_command+=" EVAL_BACKEND=$(quote "${EVAL_BACKEND}")"
run_command+=" EVAL_PROMPTS_JSONL=$(quote "${EVAL_PROMPTS_JSONL}")"
run_command+=" EVAL_PROMPTS_LIMIT=$(quote "${EVAL_PROMPTS_LIMIT}")"
run_command+=" EVAL_GAMMA=$(quote "${EVAL_GAMMA}")"
run_command+=" EVAL_MAX_NEW_TOKENS=$(quote "${EVAL_MAX_NEW_TOKENS}")"
run_command+=" EVAL_MODE=$(quote "${EVAL_MODE}")"
run_command+=" EVAL_TEMPERATURE=$(quote "${EVAL_TEMPERATURE}")"
run_command+=" EVAL_TOP_P=$(quote "${EVAL_TOP_P}")"
run_command+=" EVAL_WARMUP=$(quote "${EVAL_WARMUP}")"
run_command+=" EVAL_REPEATS=$(quote "${EVAL_REPEATS}")"
run_command+=" EVAL_REQUEST_BATCH_SIZE=$(quote "${EVAL_REQUEST_BATCH_SIZE}")"
run_command+=" EVAL_MAX_MODEL_LEN=$(quote "${EVAL_MAX_MODEL_LEN}")"
run_command+=" EVAL_GPU_MEMORY_UTILIZATION=$(quote "${EVAL_GPU_MEMORY_UTILIZATION}")"
run_command+=" EVAL_REPORT_TO_WANDB=$(quote "${EVAL_REPORT_TO_WANDB}")"
run_command+=" RUN_NAME_PREFIX=$(quote "${RUN_NAME_PREFIX}")"
run_command+=" EXPERIMENT_NAME=$(quote "${EXPERIMENT_NAME}")"
run_command+=" WANDB_GROUP=$(quote "${WANDB_GROUP}")"
run_command+=" bash scripts/run_qwen3_8b_kd_eval.sh"

echo ">>> RUN_COMMAND chars: ${#run_command}"
echo ">>> Submitting Qwen3 8B/0.6B KD+vLLM eval job: ${EXPERIMENT_NAME}"
REPO_URL="${REPO_URL}" \
REPO_BRANCH="${REPO_BRANCH}" \
REPO_DIR="${REPO_DIR}" \
CHECKPOINTS_DIR="${CHECKPOINTS_ROOT}" \
DATA_DIR="${DATA_DIR}" \
HYDRA_OUTPUTS_DIR="${HYDRA_OUTPUTS_DIR}" \
WANDB_DIR="${WANDB_DIR}" \
RUN_NAME="${EXPERIMENT_NAME}" \
RUN_COMMAND="${run_command}" \
./rcp_support/submit_train.sh "qwen3-8b-kd-eval"
