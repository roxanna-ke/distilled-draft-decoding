#!/bin/bash
# Evaluate the Qwen3-8B / Qwen3-0.6B interleaved RKL S500 checkpoint with the
# vLLM speculative-decoding backend and record acceptance / speedup metrics.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ENV_SH="${KDSD_ENV_SH:-${ROOT}/scripts/env.sh}"
source "${ENV_SH}"
echo ">>> Sourced env: ${ENV_SH}"
echo ">>> Python: ${KDSD_PYTHON}"

WORK_ROOT="${WORK_ROOT:-/scratch/cs552-mnlp-kzy}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${WORK_ROOT}/checkpoints}"
RESULTS_DIR_ROOT="${RESULTS_DIR_ROOT:-${WORK_ROOT}/results}"
WANDB_DIR="${WANDB_DIR:-${WORK_ROOT}/wandb}"

TARGET_ID="${TARGET_ID:-Qwen/Qwen3-8B}"
DRAFT_CHECKPOINT="${DRAFT_CHECKPOINT:-${CHECKPOINTS_ROOT}/qwen3_8btarget_0p6b_interleaved_rkl_s500_ultrachat_50k_a1.0_seed42/model}"
DATA_CFG="${DATA_CFG:-eval_holdout}"
SEED="${SEED:-42}"

PROMPTS_LIMIT="${PROMPTS_LIMIT:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
GAMMAS="${GAMMAS:-1 2 4}"

EVAL_MODE="${EVAL_MODE:-sampling}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.9}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-3}"
EVAL_REQUEST_BATCH_SIZE="${EVAL_REQUEST_BATCH_SIZE:-1}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-2048}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.9}"
EVAL_ENFORCE_EAGER="${EVAL_ENFORCE_EAGER:-false}"
RUN_VANILLA_BASELINE="${RUN_VANILLA_BASELINE:-true}"
WRITE_GENERATIONS="${WRITE_GENERATIONS:-true}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"

RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qwen3_8btarget_0p6b_interleaved_rkl_s500_eval}"

mkdir -p "${RESULTS_DIR_ROOT}" "${WANDB_DIR}"

if [[ ! -d "${DRAFT_CHECKPOINT}" ]]; then
  echo "ERROR: draft checkpoint not found: ${DRAFT_CHECKPOINT}" >&2
  exit 1
fi

echo ">>> Results root: ${RESULTS_DIR_ROOT}"
echo ">>> W&B dir: ${WANDB_DIR}"
echo ">>> Draft checkpoint: ${DRAFT_CHECKPOINT}"
echo ">>> Target: ${TARGET_ID}"
echo ">>> Prompt limit: ${PROMPTS_LIMIT}"
echo ">>> Gammas: ${GAMMAS}"
echo ">>> Max new tokens: ${MAX_NEW_TOKENS}"
echo ">>> vLLM batch/max_model_len/gpu_mem/eager: ${EVAL_REQUEST_BATCH_SIZE}/${EVAL_MAX_MODEL_LEN}/${EVAL_GPU_MEMORY_UTILIZATION}/${EVAL_ENFORCE_EAGER}"
echo ">>> Vanilla baseline: ${RUN_VANILLA_BASELINE}"
echo ">>> W&B enabled: ${WANDB_ENABLED}"

run_single_eval() {
  local gamma="$1"
  local run_name="${RUN_NAME_PREFIX}_g${gamma}_max${MAX_NEW_TOKENS}"
  local results_dir="${RESULTS_DIR_ROOT}/${run_name}"

  echo ">>> Eval start: run=${run_name} gamma=${gamma} max_new=${MAX_NEW_TOKENS}"

  WANDB_DIR="${WANDB_DIR}" \
  "${KDSD_PYTHON}" scripts/evaluate_sd.py \
    "model=qwen3" \
    "data=${DATA_CFG}" \
    "model.target=${TARGET_ID}" \
    "draft=${DRAFT_CHECKPOINT}" \
    "run_name=${run_name}" \
    "results_dir=${results_dir}" \
    "seed=${SEED}" \
    "prompts.hf_dataset.name=HuggingFaceH4/ultrachat_200k" \
    "prompts.hf_dataset.split=test_sft" \
    "prompts.hf_dataset.prompt_field=prompt" \
    "prompts.limit=${PROMPTS_LIMIT}" \
    "runtime.mode=${EVAL_MODE}" \
    "runtime.temperature=${EVAL_TEMPERATURE}" \
    "runtime.top_p=${EVAL_TOP_P}" \
    "runtime.gamma=${gamma}" \
    "runtime.max_new_tokens=${MAX_NEW_TOKENS}" \
    "eval.backend=vllm" \
    "eval.n_warmup=${EVAL_WARMUP}" \
    "eval.n_repeats=${EVAL_REPEATS}" \
    "eval.run_vanilla_baseline=${RUN_VANILLA_BASELINE}" \
    "eval.write_generations=${WRITE_GENERATIONS}" \
    "eval.vllm.request_batch_size=${EVAL_REQUEST_BATCH_SIZE}" \
    "eval.vllm.max_model_len=${EVAL_MAX_MODEL_LEN}" \
    "eval.vllm.gpu_memory_utilization=${EVAL_GPU_MEMORY_UTILIZATION}" \
    "eval.vllm.enforce_eager=${EVAL_ENFORCE_EAGER}" \
    "wandb.enabled=${WANDB_ENABLED}"
}

for gamma in ${GAMMAS}; do
  run_single_eval "${gamma}"
done

echo ">>> Final eval summary"
"${KDSD_PYTHON}" - "${RESULTS_DIR_ROOT}" "${RUN_NAME_PREFIX}" "${MAX_NEW_TOKENS}" ${GAMMAS} <<'PY'
import json
import math
import sys
from pathlib import Path

results_root = Path(sys.argv[1])
run_name_prefix = sys.argv[2]
max_new_tokens = sys.argv[3]
gammas = sys.argv[4:]

headers = ("run", "accept", "avg_acc", "tok/s", "sd_s", "vanilla_s", "speedup")
rows = [headers]

for gamma in gammas:
    run_name = f"{run_name_prefix}_g{gamma}_max{max_new_tokens}"
    path = results_root / run_name / "eval_summary.json"
    if not path.exists():
        rows.append((run_name, "missing", "", "", "", "", ""))
        continue

    with path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)

    speedup = summary.get("speedup")
    acceptance = summary.get("acceptance_rate")
    if not isinstance(speedup, (int, float)) or not math.isfinite(float(speedup)):
        raise SystemExit(f"Invalid speedup in {path}")
    if not isinstance(acceptance, (int, float)) or not math.isfinite(float(acceptance)):
        raise SystemExit(f"Invalid acceptance_rate in {path}")

    rows.append(
        (
            run_name,
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
