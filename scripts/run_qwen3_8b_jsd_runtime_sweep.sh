#!/bin/bash
# Run a sequential vLLM eval sweep for the Qwen3-8B target against a single
# JSD-trained 0.6B draft checkpoint.
#
# The sweep covers gamma in {1,2,4} and max_new_tokens in {64,128,256}.
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

mkdir -p "${RESULTS_DIR_ROOT}"

echo ">>> Results root: ${RESULTS_DIR_ROOT}"
echo ">>> Checkpoints root: ${CHECKPOINTS_ROOT}"
echo ">>> Target: ${TARGET_ID}"
echo ">>> Pretrained draft: ${PRETRAINED_DRAFT_ID}"
echo ">>> JSD draft: ${DRAFT_CHECKPOINT}"
echo ">>> Eval backend: ${EVAL_BACKEND}"
echo ">>> Eval mode/temp/top_p: ${EVAL_MODE}/${EVAL_TEMPERATURE}/${EVAL_TOP_P}"
echo ">>> Eval warmup/repeats: ${EVAL_WARMUP}/${EVAL_REPEATS}"
echo ">>> Eval batch/max_model_len/gpu_mem: ${EVAL_REQUEST_BATCH_SIZE}/${EVAL_MAX_MODEL_LEN}/${EVAL_GPU_MEMORY_UTILIZATION}"
echo ">>> Prompts JSONL: ${EVAL_PROMPTS_JSONL} limit=${EVAL_PROMPTS_LIMIT}"
echo ">>> Gammas: ${GAMMAS}"
echo ">>> Max new tokens: ${MAX_NEW_TOKENS_VALUES}"

if [[ ! -d "${DRAFT_CHECKPOINT}" ]]; then
  echo "ERROR: draft checkpoint not found: ${DRAFT_CHECKPOINT}" >&2
  exit 1
fi

summary_runs=()

run_single_eval() {
  local draft_spec="$1"
  local gamma="$2"
  local max_new="$3"
  local run_name="$4"
  local results_dir="$5"

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
    "eval.run_vanilla_baseline=true"
    "eval.vllm.request_batch_size=${EVAL_REQUEST_BATCH_SIZE}"
    "eval.vllm.max_model_len=${EVAL_MAX_MODEL_LEN}"
    "eval.vllm.gpu_memory_utilization=${EVAL_GPU_MEMORY_UTILIZATION}"
    "wandb.enabled=${EVAL_REPORT_TO_WANDB}"
    "results_dir=${results_dir}"
    "run_name=${run_name}"
    "seed=${SEED}"
    "prompts.jsonl=${EVAL_PROMPTS_JSONL}"
    "prompts.limit=${EVAL_PROMPTS_LIMIT}"
  )

  echo ">>> Eval start: draft=${draft_spec} gamma=${gamma} max_new=${max_new} run=${run_name}"
  "${cmd[@]}"
}

result_is_complete() {
  local results_dir="$1"
  "${KDSD_PYTHON}" - "${results_dir}" <<'PY'
import json
import math
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
path = results_dir / "eval_summary.json"
if not path.exists():
    raise SystemExit(1)
with path.open("r", encoding="utf-8") as fh:
    summary = json.load(fh)
for key in ("speedup", "acceptance_rate", "vanilla_time_s", "sd_time_s"):
    value = summary.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SystemExit(1)
PY
}

for gamma in ${GAMMAS}; do
  for max_new in ${MAX_NEW_TOKENS_VALUES}; do
    run_name="${RUN_NAME_PREFIX}_${DRAFT_LABEL}_g${gamma}_max${max_new}"
    results_dir="${RESULTS_DIR_ROOT}/${run_name}"

    if result_is_complete "${results_dir}"; then
      echo ">>> Skipping completed eval: ${run_name}"
      summary_runs+=("${run_name}")
      continue
    fi

    run_single_eval "${DRAFT_CHECKPOINT}" "${gamma}" "${max_new}" "${run_name}" "${results_dir}"
    summary_runs+=("${run_name}")
  done
done

echo ">>> Final eval sweep summary"
RESULTS_DIR_ROOT="${RESULTS_DIR_ROOT}" "${KDSD_PYTHON}" - "${summary_runs[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

results_root = Path(os.environ["RESULTS_DIR_ROOT"])
headers = ("run", "accept", "avg_acc", "tok/s", "sd_s", "vanilla_s", "speedup")
rows = [headers]

for run in sys.argv[1:]:
    path = results_root / run / "eval_summary.json"
    if not path.exists():
        rows.append((run, "missing", "", "", "", "", ""))
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
            f"{summary.get('vanilla_time_s', float('nan')):.2f}",
            f"{summary.get('speedup', float('nan')):.3f}",
        )
    )

widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
for idx, row in enumerate(rows):
    print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    if idx == 0:
        print("  ".join("-" * widths[i] for i in range(len(headers))))
PY
