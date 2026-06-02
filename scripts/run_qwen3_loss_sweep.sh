#!/bin/bash
# Run one Qwen3 loss sweep, then optionally evaluate pretrained and trained
# drafts. This script is executed inside the RunAI pod from the checked-out repo.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/env.sh"
echo ">>> Python: ${KDSD_PYTHON}"


DRAFT_SIZE="${DRAFT_SIZE:-0.6b}"  # 0.6b or 1.7b
DATA="${DATA:-ultrachat_50k}"
STEPS="${STEPS:-0}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-32}"
LR="${LR:-1e-5}"
ALPHA="${ALPHA:-1.0}"
TEMP="${TEMP:-1.0}"
SEED="${SEED:-42}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
KD_CHUNK_SIZE="${KD_CHUNK_SIZE:-128}"
COMPILE_TARGET="${COMPILE_TARGET:-false}"
TARGET_ID="${TARGET_ID:-}"
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
EVAL_BACKEND="${EVAL_BACKEND:-manual}"
EVAL_REPORT_TO_WANDB="${EVAL_REPORT_TO_WANDB:-true}"

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
    ;;
  *)
    echo "ERROR: DRAFT_SIZE must be 0.6b or 1.7b, got '${DRAFT_SIZE}'." >&2
    exit 1
    ;;
esac

for loss in ${LOSSES}; do
  if [[ "${DRAFT_TAG}" == "1p7b" && "${loss}" != "ce" && "${ALLOW_QWEN3_1_7B_FULL_KD:-0}" != "1" ]]; then
    echo "ERROR: refusing likely-OOM Qwen3-1.7B KD loss '${loss}' without ALLOW_QWEN3_1_7B_FULL_KD=1." >&2
    exit 1
  fi
done

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_${DRAFT_TAG}_${DATA}_seed${SEED}}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qwen3_${DRAFT_TAG}}"

target_override=()
if [[ -n "${TARGET_ID}" ]]; then
  target_override=("model.target=${TARGET_ID}")
fi

echo ">>> Qwen3 experiment: ${EXPERIMENT_NAME}"
echo ">>> W&B group: ${WANDB_GROUP}"
echo ">>> run name prefix: ${RUN_NAME_PREFIX}"
echo ">>> Losses: ${LOSSES}"
echo ">>> max_seq_len: ${MAX_SEQ_LEN}"
echo ">>> max_steps: ${STEPS}"
echo ">>> epochs: ${EPOCHS}"
echo ">>> per-device batch size: ${BATCH_SIZE}"
echo ">>> gradient accumulation steps: ${GRAD_ACCUM_STEPS}"
echo ">>> KD chunk size: ${KD_CHUNK_SIZE}"
echo ">>> compile_target: ${COMPILE_TARGET}"
echo ">>> target override: ${TARGET_ID:-<config default>}"
echo ">>> eval reporting steps: ${EVAL_REPORTING_STEPS}"
echo ">>> run eval after training: ${RUN_EVAL}"
echo ">>> eval pretrained baseline: ${EVAL_PRETRAINED_BASELINE}"
echo ">>> eval prompts: ${EVAL_PROMPTS_JSONL}"
echo ">>> eval prompts limit: ${EVAL_PROMPTS_LIMIT}"
echo ">>> eval gamma: ${EVAL_GAMMA}"
echo ">>> eval max_new_tokens: ${EVAL_MAX_NEW_TOKENS}"
echo ">>> eval warmup/repeats: ${EVAL_WARMUP}/${EVAL_REPEATS}"
echo ">>> eval backend: ${EVAL_BACKEND}"
echo ">>> eval report to W&B: ${EVAL_REPORT_TO_WANDB}"

export PYTORCH_CUDA_ALLOC_CONF

for loss in ${LOSSES}; do
  run_name="${RUN_NAME_PREFIX}_${loss}_${DATA}_seed${SEED}"

  if [[ "${loss}" == "ce" ]]; then
    loss_overrides=("loss=ce")
  else
    loss_overrides=("loss=${loss}" "loss.alpha=${ALPHA}" "loss.temperature=${TEMP}")
  fi

  echo ">>> Starting ${run_name}"
  WANDB_GROUP="${WANDB_GROUP}" \
  WANDB_NAME="${run_name}" \
  WANDB_JOB_TYPE="train" \
  "${KDSD_PYTHON}" scripts/train.py \
    model=qwen3 train=a100_40gb_qwen3 "data=${DATA}" "${target_override[@]}" "${loss_overrides[@]}" \
    "train.draft_init=${DRAFT_ID}" \
    "train.max_steps=${STEPS}" \
    "train.num_train_epochs=${EPOCHS}" \
    "train.per_device_train_batch_size=${BATCH_SIZE}" \
    "train.per_device_eval_batch_size=${BATCH_SIZE}" \
    "train.gradient_accumulation_steps=${GRAD_ACCUM_STEPS}" \
    "train.learning_rate=${LR}" \
    "train.compile_target=${COMPILE_TARGET}" \
    "train.eval_steps=${EVAL_REPORTING_STEPS}" \
    "data.max_seq_len=${MAX_SEQ_LEN}" \
    "loss.chunk_size=${KD_CHUNK_SIZE}" \
    "seed=${SEED}" \
    "run_name=${run_name}"

  echo ">>> Finished ${run_name}"
done

if [[ "${RUN_EVAL}" != "true" && "${RUN_EVAL}" != "1" ]]; then
  echo ">>> RUN_EVAL=${RUN_EVAL}; skipping SD evaluation"
  exit 0
fi

echo ">>> Training sweep finished; starting SD evaluation"
eval_result_runs=()

if [[ "${EVAL_PRETRAINED_BASELINE}" == "true" || "${EVAL_PRETRAINED_BASELINE}" == "1" ]]; then
  baseline_eval_run="${RUN_NAME_PREFIX}_pretrain_${DATA}_seed${SEED}_eval_g${EVAL_GAMMA}_max${EVAL_MAX_NEW_TOKENS}"
  echo ">>> Evaluating pretrained draft baseline: ${baseline_eval_run}"
  "${KDSD_PYTHON}" scripts/evaluate_sd.py \
    model=qwen3 "data=${DATA}" "${target_override[@]}" \
    "draft=${DRAFT_ID}" \
    "prompts.jsonl=${EVAL_PROMPTS_JSONL}" \
    "prompts.limit=${EVAL_PROMPTS_LIMIT}" \
    "runtime.gamma=${EVAL_GAMMA}" \
    "runtime.max_new_tokens=${EVAL_MAX_NEW_TOKENS}" \
    "eval.backend=${EVAL_BACKEND}" \
    "eval.n_warmup=${EVAL_WARMUP}" \
    "eval.n_repeats=${EVAL_REPEATS}" \
    "wandb.enabled=${EVAL_REPORT_TO_WANDB}" \
    "run_name=${baseline_eval_run}"
  eval_result_runs+=("${baseline_eval_run}")
fi

for loss in ${LOSSES}; do
  train_run="${RUN_NAME_PREFIX}_${loss}_${DATA}_seed${SEED}"
  eval_run="${train_run}_eval_g${EVAL_GAMMA}_max${EVAL_MAX_NEW_TOKENS}"

  echo ">>> Evaluating trained draft: ${eval_run}"
  "${KDSD_PYTHON}" scripts/evaluate_sd.py \
    model=qwen3 "data=${DATA}" "${target_override[@]}" \
    "draft=checkpoints/${train_run}/model" \
    "prompts.jsonl=${EVAL_PROMPTS_JSONL}" \
    "prompts.limit=${EVAL_PROMPTS_LIMIT}" \
    "runtime.gamma=${EVAL_GAMMA}" \
    "runtime.max_new_tokens=${EVAL_MAX_NEW_TOKENS}" \
    "eval.backend=${EVAL_BACKEND}" \
    "eval.n_warmup=${EVAL_WARMUP}" \
    "eval.n_repeats=${EVAL_REPEATS}" \
    "wandb.enabled=${EVAL_REPORT_TO_WANDB}" \
    "run_name=${eval_run}"
  eval_result_runs+=("${eval_run}")
done

echo ">>> Final SD evaluation summary"
"${KDSD_PYTHON}" - "${eval_result_runs[@]}" <<'PY'
import json
import sys
from pathlib import Path

rows = []
for run in sys.argv[1:]:
    path = Path("/scratch/cs552-results") / run / "eval_summary.json"
    if not path.exists():
        rows.append((run, "missing", "", "", "", "", ""))
        continue
    with path.open() as f:
        summary = json.load(f)
    rows.append(
        (
            run,
            "%.3fx" % summary.get("speedup", float("nan")),
            "%.3f" % summary.get("acceptance_rate", float("nan")),
            "%.2f" % summary.get("avg_accepted_tokens", float("nan")),
            "%.2f" % summary.get("tokens_per_second", float("nan")),
            "%.2f" % summary.get("sd_time_s", float("nan")),
            "%.2f" % summary.get("vanilla_time_s", float("nan")),
        )
    )

headers = ("run", "speedup", "accept", "avg_acc", "tok/s", "sd_s", "vanilla_s")
widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
print(line)
print("-" * len(line))
for row in rows:
    print("  ".join(str(x).ljust(w) for x, w in zip(row, widths)))
PY
