#!/bin/bash
# Run a sequential vLLM runtime sweep for the Qwen3-8B target against the
# pretrained 0.6B draft plus two selected RKL KD checkpoints.
#
# The sweep covers gamma in {1,2,4,6,8} and max_new_tokens in {128,256}.
# Each eval runs in its own Python process to avoid cross-run GPU state leaks.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ENV_SH="${KDSD_ENV_SH:-${ROOT}/scripts/env.sh}"
source "${ENV_SH}"
echo ">>> Sourced env: ${ENV_SH}"
echo ">>> Python: ${KDSD_PYTHON}"

WORK_ROOT="${WORK_ROOT:-/scratch/cs552-mnlp-kzy}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${CHECKPOINTS_DIR:-/scratch/cs552-checkpoints}}"
RESULTS_DIR_ROOT="${RESULTS_DIR_ROOT:-${WORK_ROOT}/results}"

TARGET_ID="${TARGET_ID:-Qwen/Qwen3-8B}"
PRETRAINED_DRAFT_ID="${PRETRAINED_DRAFT_ID:-Qwen/Qwen3-0.6B}"
DATA="${DATA:-ultrachat_50k}"
SEED="${SEED:-42}"

EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
EVAL_MODE="${EVAL_MODE:-greedy}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.9}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-3}"
EVAL_REQUEST_BATCH_SIZE="${EVAL_REQUEST_BATCH_SIZE:-1}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-2048}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.9}"
EVAL_REPORT_TO_WANDB="${EVAL_REPORT_TO_WANDB:-false}"
EVAL_RUN_VANILLA_BASELINE="${EVAL_RUN_VANILLA_BASELINE:-false}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-}"
EVAL_PROMPTS_LIMIT="${EVAL_PROMPTS_LIMIT:-256}"

GAMMAS="${GAMMAS:-1 2 4 6 8}"
MAX_NEW_TOKENS_VALUES="${MAX_NEW_TOKENS_VALUES:-128 256}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qwen3_8b_rkl_runtime_sweep}"

mkdir -p "${RESULTS_DIR_ROOT}"

echo ">>> Results root: ${RESULTS_DIR_ROOT}"
echo ">>> Checkpoints root: ${CHECKPOINTS_ROOT}"
echo ">>> Target: ${TARGET_ID}"
echo ">>> Eval backend: ${EVAL_BACKEND}"
echo ">>> Eval mode/temp/top_p: ${EVAL_MODE}/${EVAL_TEMPERATURE}/${EVAL_TOP_P}"
echo ">>> Eval warmup/repeats: ${EVAL_WARMUP}/${EVAL_REPEATS}"
echo ">>> Eval batch/max_model_len/gpu_mem: ${EVAL_REQUEST_BATCH_SIZE}/${EVAL_MAX_MODEL_LEN}/${EVAL_GPU_MEMORY_UTILIZATION}"
echo ">>> Gammas: ${GAMMAS}"
echo ">>> Max new tokens: ${MAX_NEW_TOKENS_VALUES}"
if [[ -n "${EVAL_PROMPTS_JSONL}" ]]; then
  echo ">>> Prompts JSONL: ${EVAL_PROMPTS_JSONL} limit=${EVAL_PROMPTS_LIMIT}"
else
  echo ">>> Prompts JSONL not set; evaluate_sd.py will fall back to its built-in smoke prompts"
fi

declare -a DRAFT_SPECS=(
  "pretrained|${PRETRAINED_DRAFT_ID}"
  "rkl_8btarget_50k|${CHECKPOINTS_ROOT}/qwen3_8btarget_0p6b_rkl_ultrachat_50k_seed42/model"
  "rkl_8btarget_tgen_50k|${CHECKPOINTS_ROOT}/qwen3_8btarget_0p6b_tgen_rkl_ultrachat_50k_target_gen_seed42/model"
)

summary_runs=()

run_single_eval() {
  local draft_label="$1"
  local draft_spec="$2"
  local gamma="$3"
  local max_new="$4"
  local run_name="$5"
  local results_dir="$6"

  local -a cmd=(
    "${KDSD_PYTHON}" scripts/evaluate_sd.py
    "model=qwen3"
    "data=${DATA}"
    "model.target=${TARGET_ID}"
    "model.draft_default=${PRETRAINED_DRAFT_ID}"
    "draft=${draft_spec}"
    "runtime.mode=${EVAL_MODE}"
    "runtime.temperature=${EVAL_TEMPERATURE}"
    "runtime.top_p=${EVAL_TOP_P}"
    "runtime.gamma=${gamma}"
    "runtime.max_new_tokens=${max_new}"
    "eval.backend=${EVAL_BACKEND}"
    "eval.n_warmup=${EVAL_WARMUP}"
    "eval.n_repeats=${EVAL_REPEATS}"
    "eval.run_vanilla_baseline=${EVAL_RUN_VANILLA_BASELINE}"
    "eval.vllm.request_batch_size=${EVAL_REQUEST_BATCH_SIZE}"
    "eval.vllm.max_model_len=${EVAL_MAX_MODEL_LEN}"
    "eval.vllm.gpu_memory_utilization=${EVAL_GPU_MEMORY_UTILIZATION}"
    "wandb.enabled=${EVAL_REPORT_TO_WANDB}"
    "results_dir=${results_dir}"
    "run_name=${run_name}"
    "seed=${SEED}"
  )

  if [[ -n "${EVAL_PROMPTS_JSONL}" ]]; then
    cmd+=("prompts.jsonl=${EVAL_PROMPTS_JSONL}")
  fi
  if [[ -n "${EVAL_PROMPTS_LIMIT}" ]]; then
    cmd+=("prompts.limit=${EVAL_PROMPTS_LIMIT}")
  fi

  echo ">>> Eval start: draft=${draft_label} gamma=${gamma} max_new=${max_new} run=${run_name}"
  "${cmd[@]}"
}

for draft_entry in "${DRAFT_SPECS[@]}"; do
  draft_label="${draft_entry%%|*}"
  draft_spec="${draft_entry#*|}"

  if [[ "${draft_label}" != "pretrained" && ! -d "${draft_spec}" ]]; then
    echo "ERROR: draft checkpoint not found: ${draft_spec}" >&2
    exit 1
  fi

  for gamma in ${GAMMAS}; do
    for max_new in ${MAX_NEW_TOKENS_VALUES}; do
      run_name="${RUN_NAME_PREFIX}_${draft_label}_g${gamma}_max${max_new}"
      results_dir="${RESULTS_DIR_ROOT}/${run_name}"
      run_single_eval "${draft_label}" "${draft_spec}" "${gamma}" "${max_new}" "${run_name}" "${results_dir}"
      summary_runs+=("${run_name}")
    done
  done
done

echo ">>> Final runtime sweep summary"
"${KDSD_PYTHON}" - "${summary_runs[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

results_root = Path(os.environ.get("RESULTS_DIR_ROOT", "/scratch/cs552-mnlp-kzy/results"))
headers = ("run", "accept", "avg_acc", "tok/s", "sd_s")
rows = [headers]

for run in sys.argv[1:]:
    path = results_root / run / "eval_summary.json"
    if not path.exists():
        rows.append((run, "missing", "", "", ""))
        continue
    with path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    rows.append(
        (
            run,
            f"{summary.get('acceptance_rate', float('nan')):.3f}",
            f"{summary.get('avg_accepted_tokens', float('nan')):.2f}",
            f"{summary.get('tokens_per_second', float('nan')):.2f}",
            f"{summary.get('sd_time_s', float('nan')):.2f}",
        )
    )

widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
for idx, row in enumerate(rows):
    print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    if idx == 0:
        print("  ".join("-" * widths[i] for i in range(len(headers))))
PY
