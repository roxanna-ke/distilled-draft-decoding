#!/bin/bash
# Submit one Run:AI job that runs one Qwen3 training run per loss with an 8B
# target. This mirrors submit_qwen3_loss_sweep.sh, but uses Qwen/Qwen3-8B as the
# KD teacher and raises the microbatch because the target is smaller than 14B.
# Override env vars as needed, e.g.
#   LOSSES="fkl rkl jsd ce" WANDB_GROUP=qwen3_8b_target_sweep ./scripts/submit_qwen3_8b_loss_sweep.sh

set -euo pipefail

DRAFT_SIZE="${DRAFT_SIZE:-0.6b}"  # 0.6b or 1.7b
DATA="${DATA:-ultrachat_50k}"
STEPS="${STEPS:-0}" # Set max_steps=0 to use num_train_epochs
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
LR="${LR:-1e-5}"
ALPHA="${ALPHA:-1.0}" # only used for non-CE losses; ignored when LOSSES includes only "ce".
TEMP="${TEMP:-1.0}"
SEED="${SEED:-42}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
KD_CHUNK_SIZE="${KD_CHUNK_SIZE:-128}"
COMPILE_TARGET="${COMPILE_TARGET:-false}"
TARGET_ID="${TARGET_ID:-Qwen/Qwen3-8B}"
EVAL_REPORTING_STEPS="${EVAL_REPORTING_STEPS:-100}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_EVAL="${RUN_EVAL:-true}"
EVAL_PRETRAINED_BASELINE="${EVAL_PRETRAINED_BASELINE:-true}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-/scratch/cs552-data/processed/${DATA}/eval.jsonl}"
EVAL_PROMPTS_LIMIT="${EVAL_PROMPTS_LIMIT:-50}"
EVAL_GAMMA="${EVAL_GAMMA:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-3}"
EVAL_REPORT_TO_WANDB="${EVAL_REPORT_TO_WANDB:-true}"

REPO_BRANCH="${REPO_BRANCH:-codex/qwen3}"

case "${DRAFT_SIZE}" in
  0.6b|0_6b)
    DRAFT_ID="Qwen/Qwen3-0.6B"
    DRAFT_TAG="0p6b"
    LOSSES="${LOSSES:-fkl rkl jsd ce}"
    ;;
  1.7b|1_7b)
    DRAFT_ID="Qwen/Qwen3-1.7B"
    DRAFT_TAG="1p7b"
    LOSSES="${LOSSES:-ce}"
    cat >&2 <<'EOF'
WARNING: Qwen3-1.7B full fine-tuning with a resident Qwen3-8B KD target may
still be memory-heavy on a 40GB A100 for fkl/rkl/jsd. This script defaults to CE
only for DRAFT_SIZE=1.7b. Set ALLOW_QWEN3_1_7B_FULL_KD=1 and LOSSES="fkl rkl jsd"
only if you intentionally want to try those full-finetune jobs.
EOF
    ;;
  *)
    echo "ERROR: DRAFT_SIZE must be 0.6b or 1.7b, got '${DRAFT_SIZE}'." >&2
    exit 1
    ;;
esac

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8btarget_${DRAFT_TAG}_${DATA}_seed${SEED}}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qwen3_8btarget_${DRAFT_TAG}}"

for loss in ${LOSSES}; do
  if [[ "${DRAFT_TAG}" == "1p7b" && "${loss}" != "ce" && "${ALLOW_QWEN3_1_7B_FULL_KD:-0}" != "1" ]]; then
    echo "ERROR: refusing likely-OOM Qwen3-1.7B KD loss '${loss}' without ALLOW_QWEN3_1_7B_FULL_KD=1." >&2
    exit 1
  fi
done

quote() {
  printf "%q" "$1"
}

run_command="DRAFT_SIZE=$(quote "${DRAFT_SIZE}")"
run_command+=" DATA=$(quote "${DATA}")"
run_command+=" STEPS=$(quote "${STEPS}")"
run_command+=" EPOCHS=$(quote "${EPOCHS}")"
run_command+=" BATCH_SIZE=$(quote "${BATCH_SIZE}")"
run_command+=" GRAD_ACCUM_STEPS=$(quote "${GRAD_ACCUM_STEPS}")"
run_command+=" LR=$(quote "${LR}")"
run_command+=" ALPHA=$(quote "${ALPHA}")"
run_command+=" TEMP=$(quote "${TEMP}")"
run_command+=" SEED=$(quote "${SEED}")"
run_command+=" MAX_SEQ_LEN=$(quote "${MAX_SEQ_LEN}")"
run_command+=" KD_CHUNK_SIZE=$(quote "${KD_CHUNK_SIZE}")"
run_command+=" COMPILE_TARGET=$(quote "${COMPILE_TARGET}")"
run_command+=" TARGET_ID=$(quote "${TARGET_ID}")"
run_command+=" EVAL_REPORTING_STEPS=$(quote "${EVAL_REPORTING_STEPS}")"
run_command+=" PYTORCH_CUDA_ALLOC_CONF=$(quote "${PYTORCH_CUDA_ALLOC_CONF}")"
run_command+=" RUN_EVAL=$(quote "${RUN_EVAL}")"
run_command+=" EVAL_PRETRAINED_BASELINE=$(quote "${EVAL_PRETRAINED_BASELINE}")"
run_command+=" EVAL_PROMPTS_JSONL=$(quote "${EVAL_PROMPTS_JSONL}")"
run_command+=" EVAL_PROMPTS_LIMIT=$(quote "${EVAL_PROMPTS_LIMIT}")"
run_command+=" EVAL_GAMMA=$(quote "${EVAL_GAMMA}")"
run_command+=" EVAL_MAX_NEW_TOKENS=$(quote "${EVAL_MAX_NEW_TOKENS}")"
run_command+=" EVAL_WARMUP=$(quote "${EVAL_WARMUP}")"
run_command+=" EVAL_REPEATS=$(quote "${EVAL_REPEATS}")"
run_command+=" EVAL_REPORT_TO_WANDB=$(quote "${EVAL_REPORT_TO_WANDB}")"
run_command+=" LOSSES=$(quote "${LOSSES}")"
run_command+=" EXPERIMENT_NAME=$(quote "${EXPERIMENT_NAME}")"
run_command+=" WANDB_GROUP=$(quote "${WANDB_GROUP}")"
run_command+=" RUN_NAME_PREFIX=$(quote "${RUN_NAME_PREFIX}")"
run_command+=" ALLOW_QWEN3_1_7B_FULL_KD=$(quote "${ALLOW_QWEN3_1_7B_FULL_KD:-0}")"
run_command+=" bash scripts/run_qwen3_loss_sweep.sh"
echo ">>> RUN_COMMAND chars: ${#run_command} (kept short for RunAI env limit)"

echo ">>> Submitting one sequential Qwen3 8B-target experiment job: ${EXPERIMENT_NAME}"
REPO_BRANCH="${REPO_BRANCH}" RUN_NAME="${EXPERIMENT_NAME}" RUN_COMMAND="${run_command}" ./rcp_support/submit_train.sh "qwen3-8btarget-${DRAFT_TAG}-sweep"
