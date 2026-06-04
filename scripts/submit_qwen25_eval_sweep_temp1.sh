#!/bin/bash
# Submit one Run:AI job that evaluates Qwen2.5 pretrained + trained checkpoints.
# Run this from the repo root on the laptop/cluster login machine where runai is
# configured. It mirrors the knowledge-distill eval sweep, but uses the
# checkpoint layout under /scratch/cs552-mnlp-kzy/checkpoints by default.

set -euo pipefail

DRAFT_SIZE="${DRAFT_SIZE:-0.5b}"  # only 0.5b is supported for this sweep
DATA="${DATA:-ultrachat_50k}"
SEED="${SEED:-42}"
TARGET_ID="${TARGET_ID:-}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/scratch/cs552-mnlp-kzy}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EVAL_PRETRAINED_BASELINE="${EVAL_PRETRAINED_BASELINE:-true}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-/scratch/cs552-data/processed/${DATA}/eval.jsonl}"
EVAL_PROMPTS_LIMIT="${EVAL_PROMPTS_LIMIT:-50}"
EVAL_GAMMA="${EVAL_GAMMA:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-3}"
EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
EVAL_MODE="${EVAL_MODE:-sampling}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.9}"
EVAL_REPORT_TO_WANDB="${EVAL_REPORT_TO_WANDB:-true}"
EVAL_REPORT_CACHED_TO_WANDB="${EVAL_REPORT_CACHED_TO_WANDB:-true}"
FORCE_RERUN="${FORCE_RERUN:-false}"

RESULTS_ROOT="${RESULTS_ROOT:-${WORKSPACE_ROOT}/results}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${WORKSPACE_ROOT}/checkpoints}"
HYDRA_ROOT="${HYDRA_ROOT:-${WORKSPACE_ROOT}/hydra/qwen25-eval-sweep-temp1}"
PRETRAINED_CHECKPOINT_ROOT="${PRETRAINED_CHECKPOINT_ROOT:-${CHECKPOINT_ROOT}/pretrained}"

REPO_BRANCH="${REPO_BRANCH:-codex/qwen3}"

case "${DRAFT_SIZE}" in
  0.5b|0_5b)
    DRAFT_TAG="0p5b"
    LOSSES="${LOSSES:-fkl rkl jsd}"
    ;;
  *)
    echo "ERROR: DRAFT_SIZE must be 0.5b, got '${DRAFT_SIZE}'." >&2
    exit 1
    ;;
esac

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_${DRAFT_TAG}_${DATA}_seed${SEED}_temp1}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qwen25_${DRAFT_TAG}_temp1}"

quote() {
  printf "%q" "$1"
}

run_command="DRAFT_SIZE=$(quote "${DRAFT_SIZE}")"
run_command+=" DATA=$(quote "${DATA}")"
run_command+=" SEED=$(quote "${SEED}")"
run_command+=" TARGET_ID=$(quote "${TARGET_ID}")"
run_command+=" WORKSPACE_ROOT=$(quote "${WORKSPACE_ROOT}")"
run_command+=" PYTORCH_CUDA_ALLOC_CONF=$(quote "${PYTORCH_CUDA_ALLOC_CONF}")"
run_command+=" EVAL_PRETRAINED_BASELINE=$(quote "${EVAL_PRETRAINED_BASELINE}")"
run_command+=" EVAL_PROMPTS_JSONL=$(quote "${EVAL_PROMPTS_JSONL}")"
run_command+=" EVAL_PROMPTS_LIMIT=$(quote "${EVAL_PROMPTS_LIMIT}")"
run_command+=" EVAL_GAMMA=$(quote "${EVAL_GAMMA}")"
run_command+=" EVAL_MAX_NEW_TOKENS=$(quote "${EVAL_MAX_NEW_TOKENS}")"
run_command+=" EVAL_WARMUP=$(quote "${EVAL_WARMUP}")"
run_command+=" EVAL_REPEATS=$(quote "${EVAL_REPEATS}")"
run_command+=" EVAL_BACKEND=$(quote "${EVAL_BACKEND}")"
run_command+=" EVAL_MODE=$(quote "${EVAL_MODE}")"
run_command+=" EVAL_TEMPERATURE=$(quote "${EVAL_TEMPERATURE}")"
run_command+=" EVAL_TOP_P=$(quote "${EVAL_TOP_P}")"
run_command+=" EVAL_REPORT_TO_WANDB=$(quote "${EVAL_REPORT_TO_WANDB}")"
run_command+=" EVAL_REPORT_CACHED_TO_WANDB=$(quote "${EVAL_REPORT_CACHED_TO_WANDB}")"
run_command+=" FORCE_RERUN=$(quote "${FORCE_RERUN}")"
run_command+=" RESULTS_ROOT=$(quote "${RESULTS_ROOT}")"
run_command+=" CHECKPOINT_ROOT=$(quote "${CHECKPOINT_ROOT}")"
run_command+=" HYDRA_ROOT=$(quote "${HYDRA_ROOT}")"
run_command+=" PRETRAINED_CHECKPOINT_ROOT=$(quote "${PRETRAINED_CHECKPOINT_ROOT}")"
run_command+=" LOSSES=$(quote "${LOSSES}")"
run_command+=" EXPERIMENT_NAME=$(quote "${EXPERIMENT_NAME}")"
run_command+=" WANDB_GROUP=$(quote "${WANDB_GROUP}")"
run_command+=" RUN_NAME_PREFIX=$(quote "${RUN_NAME_PREFIX}")"
run_command+=" bash scripts/run_qwen25_eval_sweep_temp1.sh"
echo ">>> RUN_COMMAND chars: ${#run_command} (kept short for RunAI env limit)"

echo ">>> Submitting Qwen2.5 eval-only job: ${EXPERIMENT_NAME}"
echo ">>> Workspace root inside pod: ${WORKSPACE_ROOT}"
echo ">>> Checkpoint root inside pod: ${CHECKPOINT_ROOT}"
echo ">>> Results root inside pod: ${RESULTS_ROOT}"
REPO_BRANCH="${REPO_BRANCH}" RUN_NAME="${EXPERIMENT_NAME}-eval" RUN_COMMAND="${run_command}" ./rcp_support/submit_train.sh "qwen25-${DRAFT_TAG}-eval"
