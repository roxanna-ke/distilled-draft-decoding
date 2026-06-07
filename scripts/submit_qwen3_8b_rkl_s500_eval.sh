#!/bin/bash
# Submit one RunAI job that evaluates the Qwen3-8B / Qwen3-0.6B interleaved
# RKL S500 checkpoint with the vLLM backend.

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/scratch/cs552-mnlp-kzy}"
REPO_DIR="${REPO_DIR:-${WORK_ROOT}/repos/distilled-draft-decoding-interleaved}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${WORK_ROOT}/checkpoints}"
RESULTS_DIR_ROOT="${RESULTS_DIR_ROOT:-${WORK_ROOT}/results}"
HYDRA_OUTPUTS_DIR="${HYDRA_OUTPUTS_DIR:-${WORK_ROOT}/hydra}"
WANDB_DIR="${WANDB_DIR:-${WORK_ROOT}/wandb}"
DATA_DIR="${DATA_DIR:-/scratch/cs552-data}"

TARGET_ID="${TARGET_ID:-Qwen/Qwen3-8B}"
DRAFT_CHECKPOINT="${DRAFT_CHECKPOINT:-${CHECKPOINTS_ROOT}/qwen3_8btarget_0p6b_interleaved_rkl_s500_ultrachat_50k_a1.0_seed42/model}"
DATA_CFG="${DATA_CFG:-eval_holdout}"
SEED="${SEED:-42}"

PROMPTS_LIMIT="${PROMPTS_LIMIT:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
GAMMAS="${GAMMAS:-1 2 4}"

EVAL_MODE="${EVAL_MODE:-sampling}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.9}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-3}"
EVAL_REQUEST_BATCH_SIZE="${EVAL_REQUEST_BATCH_SIZE:-1}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-2048}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.9}"
EVAL_ENFORCE_EAGER="${EVAL_ENFORCE_EAGER:-false}"
RUN_VANILLA_BASELINE="${RUN_VANILLA_BASELINE:-true}"
WRITE_GENERATIONS="${WRITE_GENERATIONS:-true}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"

RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qwen3_8btarget_0p6b_interleaved_rkl_s500_eval}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${RUN_NAME_PREFIX}}"
REPO_BRANCH="${REPO_BRANCH:-interleaved}"
REPO_URL="${REPO_URL:-https://github.com/roxanna-ke/distilled-draft-decoding.git}"

quote() {
  printf "%q" "$1"
}

run_command="WORK_ROOT=$(quote "${WORK_ROOT}")"
run_command+=" CHECKPOINTS_ROOT=$(quote "${CHECKPOINTS_ROOT}")"
run_command+=" RESULTS_DIR_ROOT=$(quote "${RESULTS_DIR_ROOT}")"
run_command+=" WANDB_DIR=$(quote "${WANDB_DIR}")"
run_command+=" TARGET_ID=$(quote "${TARGET_ID}")"
run_command+=" DRAFT_CHECKPOINT=$(quote "${DRAFT_CHECKPOINT}")"
run_command+=" DATA_CFG=$(quote "${DATA_CFG}")"
run_command+=" SEED=$(quote "${SEED}")"
run_command+=" PROMPTS_LIMIT=$(quote "${PROMPTS_LIMIT}")"
run_command+=" MAX_NEW_TOKENS=$(quote "${MAX_NEW_TOKENS}")"
run_command+=" GAMMAS=$(quote "${GAMMAS}")"
run_command+=" EVAL_MODE=$(quote "${EVAL_MODE}")"
run_command+=" EVAL_TEMPERATURE=$(quote "${EVAL_TEMPERATURE}")"
run_command+=" EVAL_TOP_P=$(quote "${EVAL_TOP_P}")"
run_command+=" EVAL_WARMUP=$(quote "${EVAL_WARMUP}")"
run_command+=" EVAL_REPEATS=$(quote "${EVAL_REPEATS}")"
run_command+=" EVAL_REQUEST_BATCH_SIZE=$(quote "${EVAL_REQUEST_BATCH_SIZE}")"
run_command+=" EVAL_MAX_MODEL_LEN=$(quote "${EVAL_MAX_MODEL_LEN}")"
run_command+=" EVAL_GPU_MEMORY_UTILIZATION=$(quote "${EVAL_GPU_MEMORY_UTILIZATION}")"
run_command+=" EVAL_ENFORCE_EAGER=$(quote "${EVAL_ENFORCE_EAGER}")"
run_command+=" RUN_VANILLA_BASELINE=$(quote "${RUN_VANILLA_BASELINE}")"
run_command+=" WRITE_GENERATIONS=$(quote "${WRITE_GENERATIONS}")"
run_command+=" WANDB_ENABLED=$(quote "${WANDB_ENABLED}")"
run_command+=" RUN_NAME_PREFIX=$(quote "${RUN_NAME_PREFIX}")"
run_command+=" bash scripts/run_qwen3_8b_rkl_s500_eval.sh"

echo ">>> RUN_COMMAND chars: ${#run_command}"
echo ">>> Submitting Qwen3 8B/0.6B interleaved RKL S500 eval job: ${EXPERIMENT_NAME}"
REPO_URL="${REPO_URL}" \
REPO_BRANCH="${REPO_BRANCH}" \
REPO_DIR="${REPO_DIR}" \
CHECKPOINTS_DIR="${CHECKPOINTS_ROOT}" \
DATA_DIR="${DATA_DIR}" \
HYDRA_OUTPUTS_DIR="${HYDRA_OUTPUTS_DIR}" \
WANDB_DIR="${WANDB_DIR}" \
RUN_NAME="${EXPERIMENT_NAME}" \
RUN_COMMAND="${run_command}" \
./rcp_support/submit_train.sh "qwen3-8b-rkl-s500-eval"
