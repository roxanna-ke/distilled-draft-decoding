#!/bin/bash
# Run a sequential vLLM runtime sweep for the Qwen3-8B target against the
# pretrained 0.6B draft plus the original-data and target-generated RKL drafts.
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
EVAL_MODE="${EVAL_MODE:-sampling}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.9}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-3}"
EVAL_REQUEST_BATCH_SIZE="${EVAL_REQUEST_BATCH_SIZE:-1}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-2048}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.9}"
EVAL_REPORT_TO_WANDB="${EVAL_REPORT_TO_WANDB:-false}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-}"
EVAL_PROMPTS_LIMIT="${EVAL_PROMPTS_LIMIT:-256}"
KEEP_INTERMEDIATE_RESULTS="${KEEP_INTERMEDIATE_RESULTS:-false}"

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
  "rkl_original_50k|${CHECKPOINTS_ROOT}/qwen3_8btarget_0p6b_rkl_ultrachat_50k_seed42/model"
  "rkl_target_generated_50k|${CHECKPOINTS_ROOT}/qwen3_8btarget_0p6b_tgen_rkl_ultrachat_50k_target_gen_seed42/model"
)

summary_runs=()

run_single_eval() {
  local draft_spec="$1"
  local gamma="$2"
  local max_new="$3"
  local run_name="$4"
  local results_dir="$5"
  local run_vanilla="$6"

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
    "eval.run_vanilla_baseline=${run_vanilla}"
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

  echo ">>> Eval start: draft=${draft_spec} gamma=${gamma} max_new=${max_new} run=${run_name} vanilla=${run_vanilla}"
  "${cmd[@]}"
}

merge_eval_results() {
  local spec_dir="$1"
  local vanilla_dir="$2"
  local final_dir="$3"

  "${KDSD_PYTHON}" - "${spec_dir}" "${vanilla_dir}" "${final_dir}" <<'PY'
import json
import math
import shutil
import sys
from pathlib import Path

spec_dir = Path(sys.argv[1])
vanilla_dir = Path(sys.argv[2])
final_dir = Path(sys.argv[3])

with (spec_dir / "eval_summary.json").open("r", encoding="utf-8") as fh:
    spec_summary = json.load(fh)
with (vanilla_dir / "eval_summary.json").open("r", encoding="utf-8") as fh:
    vanilla_summary = json.load(fh)

sd_time_s = float(spec_summary["sd_time_s"])
vanilla_time_s = float(vanilla_summary["sd_time_s"])
speedup = vanilla_time_s / sd_time_s if sd_time_s > 0 and vanilla_time_s > 0 else None
vanilla_tps = vanilla_summary.get("tokens_per_second")

spec_summary["vanilla_time_s"] = vanilla_time_s
spec_summary["speedup"] = speedup

engines = spec_summary.setdefault("engines", {})
vllm = engines.setdefault("vllm", {})
vllm["vanilla_time_s"] = vanilla_time_s
vllm["vanilla_tokens_per_second"] = vanilla_tps
vllm["speedup"] = speedup

final_dir.mkdir(parents=True, exist_ok=True)
for name in ("generations.jsonl", "timing.json", "config.yaml"):
    if name == "timing.json":
        continue
    src = spec_dir / name
    if src.exists():
        shutil.copy2(src, final_dir / name)

with (final_dir / "eval_summary.json").open("w", encoding="utf-8") as fh:
    json.dump(spec_summary, fh, indent=2, ensure_ascii=False)
with (final_dir / "timing.json").open("w", encoding="utf-8") as fh:
    json.dump(
        {
            "sd_time_s": spec_summary.get("sd_time_s"),
            "vanilla_time_s": spec_summary.get("vanilla_time_s"),
            "tokens_per_second": spec_summary.get("tokens_per_second"),
            "n_warmup": spec_summary.get("n_warmup"),
            "n_repeats": spec_summary.get("n_repeats"),
        },
        fh,
        indent=2,
        ensure_ascii=False,
    )
PY
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
speedup = summary.get("speedup")
acceptance = summary.get("acceptance_rate")
vanilla_time_s = summary.get("vanilla_time_s")
if not isinstance(speedup, (int, float)) or not math.isfinite(float(speedup)):
    raise SystemExit(1)
if not isinstance(acceptance, (int, float)) or not math.isfinite(float(acceptance)):
    raise SystemExit(1)
if not isinstance(vanilla_time_s, (int, float)) or not math.isfinite(float(vanilla_time_s)):
    raise SystemExit(1)
PY
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
      spec_results_dir="${RESULTS_DIR_ROOT}/${run_name}__spec"
      vanilla_results_dir="${RESULTS_DIR_ROOT}/${run_name}__vanilla"

      if result_is_complete "${results_dir}"; then
        echo ">>> Skipping completed eval: ${run_name}"
        summary_runs+=("${run_name}")
        continue
      fi

      run_single_eval "${draft_spec}" "${gamma}" "${max_new}" "${run_name}__spec" "${spec_results_dir}" "false"
      run_single_eval "null" "${gamma}" "${max_new}" "${run_name}__vanilla" "${vanilla_results_dir}" "false"
      merge_eval_results "${spec_results_dir}" "${vanilla_results_dir}" "${results_dir}"
      if [[ "${KEEP_INTERMEDIATE_RESULTS}" != "true" && "${KEEP_INTERMEDIATE_RESULTS}" != "1" ]]; then
        rm -rf "${spec_results_dir}" "${vanilla_results_dir}"
      fi
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
headers = ("run", "accept", "avg_acc", "tok/s", "sd_s", "vanilla_s", "speedup")
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
