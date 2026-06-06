#!/bin/bash
# Run Qwen3-8B / Qwen3-0.6B KD loss sweep, evaluating each draft immediately
# after its training completes.
# This script is intended to run inside the RunAI pod from the checked-out repo.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ENV_SH="${KDSD_ENV_SH:-${ROOT}/scripts/env.sh}"
source "${ENV_SH}"
echo ">>> Sourced env: ${ENV_SH}"
echo ">>> Python: ${KDSD_PYTHON}"

WORK_ROOT="${WORK_ROOT:-/scratch/cs552-mnlp-kzy}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${CHECKPOINTS_DIR:-${WORK_ROOT}/checkpoints}}"
RESULTS_DIR_ROOT="${RESULTS_DIR_ROOT:-${WORK_ROOT}/results}"

DATA="${DATA:-ultrachat_50k}"
LOSSES="${LOSSES:-fkl rkl jsd}"
TARGET_ID="${TARGET_ID:-Qwen/Qwen3-8B}"
DRAFT_ID="${DRAFT_ID:-Qwen/Qwen3-0.6B}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qwen3_8btarget_0p6b_interleaved}"
SEED="${SEED:-42}"

STEPS="${STEPS:-500}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
LR="${LR:-1e-5}"
ALPHA="${ALPHA:-1.0}"
TEMP="${TEMP:-1.0}"
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

if [[ "${STEPS}" =~ ^[0-9]+$ ]] && [[ "${STEPS}" -gt 0 ]]; then
  STEP_TAG="s${STEPS}"
else
  STEP_TAG="e${EPOCHS}"
fi
RUN_NAME_PREFIX_TAGGED="${RUN_NAME_PREFIX}_${STEP_TAG}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${RUN_NAME_PREFIX_TAGGED}_${DATA}_seed${SEED}}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"

echo ">>> Experiment: ${EXPERIMENT_NAME}"
echo ">>> Work root: ${WORK_ROOT}"
echo ">>> Checkpoints root: ${CHECKPOINTS_ROOT}"
echo ">>> Eval results root: ${RESULTS_DIR_ROOT}"
echo ">>> Target: ${TARGET_ID}"
echo ">>> Draft init: ${DRAFT_ID}"
echo ">>> Data: ${DATA}"
echo ">>> Losses: ${LOSSES}"
echo ">>> alpha/temp: ${ALPHA}/${TEMP}"
echo ">>> train steps/epochs: ${STEPS}/${EPOCHS}"
echo ">>> train batch/grad accum: ${BATCH_SIZE}/${GRAD_ACCUM_STEPS}"
echo ">>> train rollin: ${TRAIN_ROLLIN}"
echo ">>> interleaved rollout/top-k: ${INTERLEAVED_ROLLOUT_TOKENS}/${INTERLEAVED_TEACHER_TOPK}"
echo ">>> eval backend: ${EVAL_BACKEND}"
echo ">>> eval prompts: ${EVAL_PROMPTS_JSONL} limit=${EVAL_PROMPTS_LIMIT}"
echo ">>> eval mode/temp/top_p: ${EVAL_MODE}/${EVAL_TEMPERATURE}/${EVAL_TOP_P}"
echo ">>> eval gamma/max_new: ${EVAL_GAMMA}/${EVAL_MAX_NEW_TOKENS}"

export PYTORCH_CUDA_ALLOC_CONF
export WORK_ROOT CHECKPOINTS_ROOT RESULTS_DIR_ROOT
mkdir -p "${CHECKPOINTS_ROOT}" "${RESULTS_DIR_ROOT}"

verify_vllm_eval() {
  local summary_path="$1"
  local expected_draft="$2"
  "${KDSD_PYTHON}" - "${summary_path}" "${expected_draft}" <<'PY'
import json
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected_draft = sys.argv[2].lower() in {"1", "true", "yes"}
with summary_path.open("r", encoding="utf-8") as fh:
    summary = json.load(fh)

engines = summary.get("engines") or {}
vllm = engines.get("vllm")
if not isinstance(vllm, dict):
    raise SystemExit(f"{summary_path}: expected engines.vllm in eval summary")

speedup = float(summary.get("speedup", float("nan")))
acceptance = float(summary.get("acceptance_rate", float("nan")))
if not math.isfinite(speedup) or not math.isfinite(acceptance):
    raise SystemExit(f"{summary_path}: non-finite speedup/acceptance_rate")

if expected_draft:
    if int(vllm.get("num_draft_tokens", 0)) <= 0:
        raise SystemExit(f"{summary_path}: vLLM ran without speculative draft tokens")
    if int(vllm.get("num_drafts", 0)) <= 0:
        raise SystemExit(f"{summary_path}: vLLM summary shows zero draft steps")
PY
}

eval_result_runs=()

if [[ ("${RUN_EVAL}" == "true" || "${RUN_EVAL}" == "1") && ("${EVAL_PRETRAINED_BASELINE}" == "true" || "${EVAL_PRETRAINED_BASELINE}" == "1") ]]; then
  baseline_eval_run="${RUN_NAME_PREFIX_TAGGED}_pretrain_${DATA}_seed${SEED}_vllm_g${EVAL_GAMMA}_max${EVAL_MAX_NEW_TOKENS}"
  baseline_results_dir="${RESULTS_DIR_ROOT}/${baseline_eval_run}"
  echo ">>> Evaluating pretrained draft baseline: ${baseline_eval_run}"
  "${KDSD_PYTHON}" scripts/evaluate_sd.py \
    model=qwen3 "data=${DATA}" \
    "model.target=${TARGET_ID}" \
    "model.draft_default=${DRAFT_ID}" \
    "draft=${DRAFT_ID}" \
    "prompts.jsonl=${EVAL_PROMPTS_JSONL}" \
    "prompts.limit=${EVAL_PROMPTS_LIMIT}" \
    "runtime.mode=${EVAL_MODE}" \
    "runtime.temperature=${EVAL_TEMPERATURE}" \
    "runtime.top_p=${EVAL_TOP_P}" \
    "runtime.gamma=${EVAL_GAMMA}" \
    "runtime.max_new_tokens=${EVAL_MAX_NEW_TOKENS}" \
    "eval.backend=${EVAL_BACKEND}" \
    "eval.n_warmup=${EVAL_WARMUP}" \
    "eval.n_repeats=${EVAL_REPEATS}" \
    "eval.vllm.request_batch_size=${EVAL_REQUEST_BATCH_SIZE}" \
    "eval.vllm.max_model_len=${EVAL_MAX_MODEL_LEN}" \
    "eval.vllm.gpu_memory_utilization=${EVAL_GPU_MEMORY_UTILIZATION}" \
    "wandb.enabled=${EVAL_REPORT_TO_WANDB}" \
    "results_dir=${baseline_results_dir}" \
    "run_name=${baseline_eval_run}"
  verify_vllm_eval "${baseline_results_dir}/eval_summary.json" true
  eval_result_runs+=("${baseline_eval_run}")
fi

for loss in ${LOSSES}; do
  run_name="${RUN_NAME_PREFIX_TAGGED}_${loss}_${DATA}_a${ALPHA}_seed${SEED}"
  checkpoint_dir="${CHECKPOINTS_ROOT}/${run_name}"
  echo ">>> Starting training: ${run_name}"

  WANDB_GROUP="${WANDB_GROUP}" \
  WANDB_NAME="${run_name}" \
  WANDB_JOB_TYPE="train" \
  "${KDSD_PYTHON}" scripts/train.py \
    model=qwen3 train=a100_40gb_qwen3 "data=${DATA}" "loss=${loss}" \
    "model.target=${TARGET_ID}" \
    "model.draft_default=${DRAFT_ID}" \
    "train.draft_init=${DRAFT_ID}" \
    "train.max_steps=${STEPS}" \
    "train.num_train_epochs=${EPOCHS}" \
    "train.per_device_train_batch_size=${BATCH_SIZE}" \
    "train.per_device_eval_batch_size=${BATCH_SIZE}" \
    "train.gradient_accumulation_steps=${GRAD_ACCUM_STEPS}" \
    "train.learning_rate=${LR}" \
    "train.compile_target=${COMPILE_TARGET}" \
    "train.eval_steps=${EVAL_REPORTING_STEPS}" \
    "train.rollin=${TRAIN_ROLLIN}" \
    "train.interleaved_rollout_tokens=${INTERLEAVED_ROLLOUT_TOKENS}" \
    "train.interleaved_student_mode=${INTERLEAVED_STUDENT_MODE}" \
    "train.interleaved_student_temperature=${INTERLEAVED_STUDENT_TEMPERATURE}" \
    "train.interleaved_student_top_p=${INTERLEAVED_STUDENT_TOP_P}" \
    "train.interleaved_teacher_topk=${INTERLEAVED_TEACHER_TOPK}" \
    "data.max_seq_len=${MAX_SEQ_LEN}" \
    "loss.alpha=${ALPHA}" \
    "loss.temperature=${TEMP}" \
    "loss.chunk_size=${KD_CHUNK_SIZE}" \
    "seed=${SEED}" \
    "output_dir=${checkpoint_dir}" \
    "run_name=${run_name}"

  echo ">>> Finished training: ${run_name}"

  if [[ "${RUN_EVAL}" != "true" && "${RUN_EVAL}" != "1" ]]; then
    continue
  fi

  eval_run="${run_name}_vllm_g${EVAL_GAMMA}_max${EVAL_MAX_NEW_TOKENS}"
  eval_results_dir="${RESULTS_DIR_ROOT}/${eval_run}"

  echo ">>> Evaluating trained draft: ${eval_run}"
  "${KDSD_PYTHON}" scripts/evaluate_sd.py \
    model=qwen3 "data=${DATA}" \
    "model.target=${TARGET_ID}" \
    "model.draft_default=${DRAFT_ID}" \
    "draft=${checkpoint_dir}/model" \
    "prompts.jsonl=${EVAL_PROMPTS_JSONL}" \
    "prompts.limit=${EVAL_PROMPTS_LIMIT}" \
    "runtime.mode=${EVAL_MODE}" \
    "runtime.temperature=${EVAL_TEMPERATURE}" \
    "runtime.top_p=${EVAL_TOP_P}" \
    "runtime.gamma=${EVAL_GAMMA}" \
    "runtime.max_new_tokens=${EVAL_MAX_NEW_TOKENS}" \
    "eval.backend=${EVAL_BACKEND}" \
    "eval.n_warmup=${EVAL_WARMUP}" \
    "eval.n_repeats=${EVAL_REPEATS}" \
    "eval.vllm.request_batch_size=${EVAL_REQUEST_BATCH_SIZE}" \
    "eval.vllm.max_model_len=${EVAL_MAX_MODEL_LEN}" \
    "eval.vllm.gpu_memory_utilization=${EVAL_GPU_MEMORY_UTILIZATION}" \
    "wandb.enabled=${EVAL_REPORT_TO_WANDB}" \
    "results_dir=${eval_results_dir}" \
    "run_name=${eval_run}"
  verify_vllm_eval "${eval_results_dir}/eval_summary.json" true
  eval_result_runs+=("${eval_run}")
done

if [[ "${RUN_EVAL}" != "true" && "${RUN_EVAL}" != "1" ]]; then
  echo ">>> RUN_EVAL=${RUN_EVAL}; skipping SD evaluation"
  exit 0
fi

echo ">>> Final SD evaluation summary"
"${KDSD_PYTHON}" - "${eval_result_runs[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

rows = []
results_root = Path(os.environ.get("RESULTS_DIR_ROOT", "/scratch/cs552-results"))
for run in sys.argv[1:]:
    path = results_root / run / "eval_summary.json"
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
