#!/bin/bash
# Submit a non-interactive Run:AI job that evaluates four draft variants with
# the vLLM backend: pretrained, fkl, rkl, jsd.
#
# This script runs locally and delegates the actual pod submission to
# rcp_support/submit_train.sh. Even if your current local checkout is on the
# train branch, the remote job will fetch and run the vllm-eval branch by
# default.

set -euo pipefail

# ============== EDIT / OVERRIDE THESE LINES ==============
GASPAR="${GASPAR:-ke}"                  # Your EPFL GASPAR username.
GROUP="${GROUP:-g67}"                       # Your team, e.g. g07.
REPO_URL="${REPO_URL:-https://github.com/roxanna-ke/distilled-draft-decoding.git}"
REPO_BRANCH="${REPO_BRANCH:-vllm-eval}"     # Remote branch used inside the pod.
WANDB_MODE="${WANDB_MODE:-online}"        # disabled | offline | online

# Legacy workspace layout used by the earlier training/eval pod scripts.
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/scratch/cs552-mnlp-kzy}"
REPO_DIR="${REPO_DIR:-${WORKSPACE_ROOT}/repos/distilled-draft-decoding-train}"
HF_HOME_DIR="${HF_HOME_DIR:-${WORKSPACE_ROOT}/hf_cache}"
RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_ROOT}/results}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${WORKSPACE_ROOT}/checkpoints}"
DATA_DIR="${DATA_DIR:-${WORKSPACE_ROOT}/data}"
HYDRA_OUTPUTS_DIR="${HYDRA_OUTPUTS_DIR:-${WORKSPACE_ROOT}/hydra}"
WANDB_DIR="${WANDB_DIR:-${WORKSPACE_ROOT}/wandb}"

# Checkpoint naming convention matches rcp_support/eval_ep_loop.sh:
#   ${CHECKPOINTS_DIR}/{fkl,rkl,jsd}_${RUN_NAME_SUFFIX}/model
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-ultra50k_s8000_seq512_a1_temp2}"

# Eval dataset / decoding config.
DATASET="${DATASET:-ultrachat_50k}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-${DATA_DIR}/processed/${DATASET}/eval.jsonl}"
EVAL_PROMPT_LIMIT="${EVAL_PROMPT_LIMIT:-50}"
EVAL_N_WARMUP="${EVAL_N_WARMUP:-0}"
EVAL_N_REPEATS="${EVAL_N_REPEATS:-1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_GAMMA="${EVAL_GAMMA:-4}"
EVAL_MODE="${EVAL_MODE:-sampling}"          # greedy | sampling
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.9}"
EVAL_RUN_VANILLA_BASELINE="${EVAL_RUN_VANILLA_BASELINE:-true}"

# Model / vLLM settings.
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
VLLM_REQUEST_BATCH_SIZE="${VLLM_REQUEST_BATCH_SIZE:-8}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
VLLM_DRAFT_TP_SIZE="${VLLM_DRAFT_TP_SIZE:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_SWAP_SPACE="${VLLM_SWAP_SPACE:-0}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"
VLLM_DISABLE_LOG_STATS="${VLLM_DISABLE_LOG_STATS:-false}"

# Draft set to evaluate. Keep the default ordering unless you have a reason to change it.
EVAL_DRAFTS="${EVAL_DRAFTS:-pretrained,fkl,rkl,jsd}"
# ========================================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
  echo "ERROR: set GASPAR to your EPFL GASPAR username." >&2
  exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
  echo "ERROR: set GROUP to your team number (e.g. g07)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_TRAIN="${SCRIPT_DIR}/submit_train.sh"

if [[ ! -x "${SUBMIT_TRAIN}" ]]; then
  echo "ERROR: missing executable submit helper: ${SUBMIT_TRAIN}" >&2
  exit 1
fi

RUN_COMMAND="bash rcp_support/run_vllm_eval_sweep.sh"

export GASPAR
export GROUP
export REPO_URL
export REPO_BRANCH
export WANDB_MODE
export WORKSPACE_ROOT
export REPO_DIR
export HF_HOME_DIR
export RESULTS_DIR
export CHECKPOINTS_DIR
export DATA_DIR
export HYDRA_OUTPUTS_DIR
export WANDB_DIR
export RUN_NAME="eval_${RUN_NAME_SUFFIX}_vllm"
export RUN_COMMAND
export RUN_NAME_SUFFIX
export DATASET
export EVAL_PROMPTS_JSONL
export EVAL_PROMPT_LIMIT
export EVAL_N_WARMUP
export EVAL_N_REPEATS
export EVAL_MAX_NEW_TOKENS
export EVAL_GAMMA
export EVAL_MODE
export EVAL_TEMPERATURE
export EVAL_TOP_P
export EVAL_RUN_VANILLA_BASELINE
export TARGET_MODEL
export DRAFT_MODEL
export MODEL_DTYPE
export VLLM_REQUEST_BATCH_SIZE
export VLLM_TP_SIZE
export VLLM_DRAFT_TP_SIZE
export VLLM_MAX_MODEL_LEN
export VLLM_GPU_MEMORY_UTILIZATION
export VLLM_SWAP_SPACE
export VLLM_ENFORCE_EAGER
export VLLM_DISABLE_LOG_STATS
export EVAL_DRAFTS

exec "${SUBMIT_TRAIN}" "${1:-vllm-eval}"
