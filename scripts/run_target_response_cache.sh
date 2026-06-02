#!/bin/bash
# Run target response cache generation inside the RunAI job checkout.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/env.sh"
echo ">>> Python: ${KDSD_PYTHON}"


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
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cache_${MODEL}_${DATA}_seed${SEED}}"

TARGET_OVERRIDE=()
if [[ -n "${TARGET_ID}" ]]; then
  TARGET_OVERRIDE=("model.target=${TARGET_ID}")
fi

LIMIT_OVERRIDE=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_OVERRIDE=("+data.limit=${LIMIT}")
fi

echo ">>> Target response cache job: ${EXPERIMENT_NAME}"
echo ">>> model config: ${MODEL}"
echo ">>> target override: ${TARGET_ID:-<config default>}"
echo ">>> data: ${DATA}"
echo ">>> splits: ${SPLITS}"
echo ">>> prepare data first: ${PREPARE_DATA}"
echo ">>> backend: ${BACKEND}"
echo ">>> request batch size: ${REQUEST_BATCH_SIZE}"
echo ">>> max new tokens: ${MAX_NEW_TOKENS}"
echo ">>> mode/temp/top_p: ${MODE}/${TEMPERATURE}/${TOP_P}"
echo ">>> max_model_len: ${MAX_MODEL_LEN}"
echo ">>> max prompt tokens: ${MAX_PROMPT_TOKENS:-<auto: max_model_len - max_new_tokens>}"
echo ">>> gpu memory utilization: ${GPU_MEMORY_UTILIZATION}"

MAX_PROMPT_TOKENS_OVERRIDE=()
if [[ -n "${MAX_PROMPT_TOKENS}" ]]; then
  MAX_PROMPT_TOKENS_OVERRIDE=("data.target_generation.max_prompt_tokens=${MAX_PROMPT_TOKENS}")
fi

if [[ "${PREPARE_DATA}" == "true" || "${PREPARE_DATA}" == "1" ]]; then
  echo ">>> Preparing processed data for ${DATA}"
  "${KDSD_PYTHON}" scripts/prepare_data.py \
    "model=${MODEL}" "data=${DATA}" "${TARGET_OVERRIDE[@]}" "${LIMIT_OVERRIDE[@]}" "seed=${SEED}"
else
  echo ">>> PREPARE_DATA=${PREPARE_DATA}; skipping prepare_data.py"
fi

echo ">>> Generating target responses"
"${KDSD_PYTHON}" scripts/generate_target_responses.py \
  "model=${MODEL}" "data=${DATA}" "${TARGET_OVERRIDE[@]}" "${LIMIT_OVERRIDE[@]}" "seed=${SEED}" \
  "data.target_generation.splits=${SPLITS}" \
  "data.target_generation.backend=${BACKEND}" \
  "data.target_generation.request_batch_size=${REQUEST_BATCH_SIZE}" \
  "data.target_generation.max_new_tokens=${MAX_NEW_TOKENS}" \
  "data.target_generation.mode=${MODE}" \
  "data.target_generation.temperature=${TEMPERATURE}" \
  "data.target_generation.top_p=${TOP_P}" \
  "data.target_generation.max_model_len=${MAX_MODEL_LEN}" \
  "${MAX_PROMPT_TOKENS_OVERRIDE[@]}" \
  "data.target_generation.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}" \
  "data.target_generation.tensor_parallel_size=${TENSOR_PARALLEL_SIZE}" \
  "data.target_generation.swap_space=${SWAP_SPACE}" \
  "data.target_generation.enforce_eager=${ENFORCE_EAGER}"

echo ">>> Target response cache finished"
