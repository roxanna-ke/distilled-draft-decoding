#!/bin/bash
# Submit one RunAI job that runs the Qwen3-8B JSD eval sweep sequentially.

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/scratch/cs552-mnlp-kzy}"
REPO_DIR="${REPO_DIR:-${WORK_ROOT}/repos/distilled-draft-decoding}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-/scratch/cs552-checkpoints}"
RESULTS_DIR_ROOT="${RESULTS_DIR_ROOT:-${WORK_ROOT}/results}"
HYDRA_OUTPUTS_DIR="${HYDRA_OUTPUTS_DIR:-${WORK_ROOT}/hydra}"
WANDB_DIR="${WANDB_DIR:-${WORK_ROOT}/wandb}"
DATA_DIR="${DATA_DIR:-/scratch/cs552-data}"

TARGET_ID="${TARGET_ID:-Qwen/Qwen3-8B}"
PRETRAINED_DRAFT_ID="${PRETRAINED_DRAFT_ID:-Qwen/Qwen3-0.6B}"
DRAFT_LABEL="${DRAFT_LABEL:-jsd_target_generated_50k}"
DRAFT_CHECKPOINT="${DRAFT_CHECKPOINT:-${CHECKPOINTS_ROOT}/qwen3_8btarget_0p6b_tgen_jsd_ultrachat_50k_target_gen_seed42/model}"
DATA="${DATA:-ultrachat_50k}"
SEED="${SEED:-42}"

EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
EVAL_MODE="${EVAL_MODE:-sampling}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.9}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-3}"
EVAL_REQUEST_BATCH_SIZE="${EVAL_REQUEST_BATCH_SIZE:-1}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-2048}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.9}"
EVAL_REPORT_TO_WANDB="${EVAL_REPORT_TO_WANDB:-false}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-/scratch/cs552-data/processed/ultrachat_50k/eval.jsonl}"
EVAL_PROMPTS_LIMIT="${EVAL_PROMPTS_LIMIT:-50}"
KEEP_INTERMEDIATE_RESULTS="${KEEP_INTERMEDIATE_RESULTS:-false}"

GAMMAS="${GAMMAS:-1 2 4}"
MAX_NEW_TOKENS_VALUES="${MAX_NEW_TOKENS_VALUES:-64 128 256}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qwen3_8b_jsd_runtime_sweep}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${RUN_NAME_PREFIX}_${DATA}_seed${SEED}}"
REPO_BRANCH="${REPO_BRANCH:-main}"
REPO_URL="${REPO_URL:-https://github.com/roxanna-ke/distilled-draft-decoding.git}"

quote() {
  printf "%q" "$1"
}

run_command="WORK_ROOT=$(quote "${WORK_ROOT}")"
run_command+=" CHECKPOINTS_ROOT=$(quote "${CHECKPOINTS_ROOT}")"
run_command+=" RESULTS_DIR_ROOT=$(quote "${RESULTS_DIR_ROOT}")"
run_command+=" TARGET_ID=$(quote "${TARGET_ID}")"
run_command+=" PRETRAINED_DRAFT_ID=$(quote "${PRETRAINED_DRAFT_ID}")"
run_command+=" DRAFT_LABEL=$(quote "${DRAFT_LABEL}")"
run_command+=" DRAFT_CHECKPOINT=$(quote "${DRAFT_CHECKPOINT}")"
run_command+=" DATA=$(quote "${DATA}")"
run_command+=" SEED=$(quote "${SEED}")"
run_command+=" EVAL_BACKEND=$(quote "${EVAL_BACKEND}")"
run_command+=" EVAL_MODE=$(quote "${EVAL_MODE}")"
run_command+=" EVAL_TEMPERATURE=$(quote "${EVAL_TEMPERATURE}")"
run_command+=" EVAL_TOP_P=$(quote "${EVAL_TOP_P}")"
run_command+=" EVAL_WARMUP=$(quote "${EVAL_WARMUP}")"
run_command+=" EVAL_REPEATS=$(quote "${EVAL_REPEATS}")"
run_command+=" EVAL_REQUEST_BATCH_SIZE=$(quote "${EVAL_REQUEST_BATCH_SIZE}")"
run_command+=" EVAL_MAX_MODEL_LEN=$(quote "${EVAL_MAX_MODEL_LEN}")"
run_command+=" EVAL_GPU_MEMORY_UTILIZATION=$(quote "${EVAL_GPU_MEMORY_UTILIZATION}")"
run_command+=" EVAL_REPORT_TO_WANDB=$(quote "${EVAL_REPORT_TO_WANDB}")"
run_command+=" EVAL_PROMPTS_JSONL=$(quote "${EVAL_PROMPTS_JSONL}")"
run_command+=" EVAL_PROMPTS_LIMIT=$(quote "${EVAL_PROMPTS_LIMIT}")"
run_command+=" KEEP_INTERMEDIATE_RESULTS=$(quote "${KEEP_INTERMEDIATE_RESULTS}")"
run_command+=" GAMMAS=$(quote "${GAMMAS}")"
run_command+=" MAX_NEW_TOKENS_VALUES=$(quote "${MAX_NEW_TOKENS_VALUES}")"
run_command+=" RUN_NAME_PREFIX=$(quote "${RUN_NAME_PREFIX}")"
run_command+=" bash scripts/run_qwen3_8b_jsd_runtime_sweep.sh"

echo ">>> RUN_COMMAND chars: ${#run_command}"
echo ">>> Submitting Qwen3-8B JSD eval sweep: ${EXPERIMENT_NAME}"
REPO_URL="${REPO_URL}" \
REPO_BRANCH="${REPO_BRANCH}" \
REPO_DIR="${REPO_DIR}" \
CHECKPOINTS_DIR="${CHECKPOINTS_ROOT}" \
DATA_DIR="${DATA_DIR}" \
HYDRA_OUTPUTS_DIR="${HYDRA_OUTPUTS_DIR}" \
WANDB_DIR="${WANDB_DIR}" \
RUN_NAME="${EXPERIMENT_NAME}" \
RUN_COMMAND="${run_command}" \
./rcp_support/submit_train.sh "qwen3-8b-jsd-sweep"
