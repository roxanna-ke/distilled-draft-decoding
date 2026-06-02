#!/bin/bash
# Submit one Run:AI job to cache target-generated responses with vLLM.
#
# Run from the repo root on a machine with runai configured. Defaults target the
# Qwen2.5 UltraChat-50k response cache on one 40GB A100. Override env vars, e.g.
#   DATA=ultrachat_10k REQUEST_BATCH_SIZE=512 ./scripts/submit_target_response_cache.sh
#   MODEL=qwen3 TARGET_ID=Qwen/Qwen3-14B DATA=ultrachat_50k ./scripts/submit_target_response_cache.sh

set -euo pipefail

MODEL="${MODEL:-qwen25}"
DATA="${DATA:-ultrachat_50k}"
TARGET_ID="${TARGET_ID:-}"
SEED="${SEED:-42}"
SPLITS="${SPLITS:-[train,val]}"
PREPARE_DATA="${PREPARE_DATA:-true}"
LIMIT="${LIMIT:-}"

BACKEND="${BACKEND:-vllm}"
REQUEST_BATCH_SIZE="${REQUEST_BATCH_SIZE:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MODE="${MODE:-greedy}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SWAP_SPACE="${SWAP_SPACE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"

REPO_BRANCH="${REPO_BRANCH:-codex/vllm-generate}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cache_${MODEL}_${DATA}_seed${SEED}}"
WANDB_MODE="${WANDB_MODE:-disabled}"

quote() {
  printf "%q" "$1"
}

run_command="MODEL=$(quote "${MODEL}")"
run_command+=" DATA=$(quote "${DATA}")"
run_command+=" TARGET_ID=$(quote "${TARGET_ID}")"
run_command+=" SEED=$(quote "${SEED}")"
run_command+=" SPLITS=$(quote "${SPLITS}")"
run_command+=" PREPARE_DATA=$(quote "${PREPARE_DATA}")"
run_command+=" LIMIT=$(quote "${LIMIT}")"
run_command+=" BACKEND=$(quote "${BACKEND}")"
run_command+=" REQUEST_BATCH_SIZE=$(quote "${REQUEST_BATCH_SIZE}")"
run_command+=" MAX_NEW_TOKENS=$(quote "${MAX_NEW_TOKENS}")"
run_command+=" MODE=$(quote "${MODE}")"
run_command+=" TEMPERATURE=$(quote "${TEMPERATURE}")"
run_command+=" TOP_P=$(quote "${TOP_P}")"
run_command+=" MAX_MODEL_LEN=$(quote "${MAX_MODEL_LEN}")"
run_command+=" MAX_PROMPT_TOKENS=$(quote "${MAX_PROMPT_TOKENS}")"
run_command+=" GPU_MEMORY_UTILIZATION=$(quote "${GPU_MEMORY_UTILIZATION}")"
run_command+=" TENSOR_PARALLEL_SIZE=$(quote "${TENSOR_PARALLEL_SIZE}")"
run_command+=" SWAP_SPACE=$(quote "${SWAP_SPACE}")"
run_command+=" ENFORCE_EAGER=$(quote "${ENFORCE_EAGER}")"
run_command+=" EXPERIMENT_NAME=$(quote "${EXPERIMENT_NAME}")"
run_command+=" bash scripts/run_target_response_cache.sh"

echo ">>> Submitting target response cache job: ${EXPERIMENT_NAME}"
echo ">>> RUN_COMMAND chars: ${#run_command} (kept short for RunAI env limit)"
REPO_BRANCH="${REPO_BRANCH}" \
RUN_NAME="${EXPERIMENT_NAME}" \
WANDB_MODE="${WANDB_MODE}" \
RUN_COMMAND="${run_command}" \
./rcp_support/submit_train.sh "tcache-${MODEL}"
