#!/bin/bash
# Evaluate Qwen2.5 pretrained + checkpoint drafts for an existing checkpoint set.
# This script runs inside the RunAI pod from the checked-out repo.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/env.sh"
echo ">>> Python: ${KDSD_PYTHON}"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/scratch/cs552-mnlp-kzy}"

DRAFT_SIZE="${DRAFT_SIZE:-0.5b}"  # only 0.5b is supported for this sweep
DATA="${DATA:-ultrachat_50k}"
SEED="${SEED:-42}"
TARGET_ID="${TARGET_ID:-}"
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

case "${DRAFT_SIZE}" in
  0.5b|0_5b)
    DRAFT_ID="Qwen/Qwen2.5-0.5B-Instruct"
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

target_override=()
if [[ -n "${TARGET_ID}" ]]; then
  target_override=("model.target=${TARGET_ID}")
fi

is_true() {
  [[ "$1" == "true" || "$1" == "1" || "$1" == "yes" || "$1" == "y" ]]
}

summary_path() {
  printf '%s/%s/eval_summary.json\n' "${RESULTS_ROOT}" "$1"
}

checkpoint_done() {
  local model_dir="$1"
  [[ -f "${model_dir}/config.json" ]] && {
    compgen -G "${model_dir}/*.safetensors" >/dev/null || \
    compgen -G "${model_dir}/pytorch_model*.bin" >/dev/null || \
    [[ -f "${model_dir}/model.safetensors.index.json" ]] || \
    [[ -f "${model_dir}/pytorch_model.bin.index.json" ]]
  }
}

checkpoint_dir_for_loss() {
  local loss="$1"
  case "${loss}" in
    fkl)
      printf '%s\n' "${CHECKPOINT_ROOT}/fkl_ultra50k_s8000_seq512_a1_temp2/model"
      ;;
    rkl)
      printf '%s\n' "${CHECKPOINT_ROOT}/rkl_ultra50k_s8000_seq512_a1_temp2/model"
      ;;
    jsd)
      printf '%s\n' "${CHECKPOINT_ROOT}/jsd_ultra50k_s8000_seq512_a1_temp2/model"
      ;;
    ce)
      printf '%s\n' "${CHECKPOINT_ROOT}/kd_ce_ultrachat_10k_targetgen_bf16_seed42/model"
      ;;
    *)
      echo "ERROR: unsupported loss '${loss}'. Override checkpoint_dir_for_loss if needed." >&2
      return 1
      ;;
  esac
}

report_cached_eval_to_wandb() {
  local eval_run="$1"
  local summary="$2"

  if ! is_true "${EVAL_REPORT_TO_WANDB}" || ! is_true "${EVAL_REPORT_CACHED_TO_WANDB}"; then
    return
  fi

  echo ">>> Reporting cached ${eval_run} metrics to W&B"
  WANDB_GROUP="${WANDB_GROUP}" \
  WANDB_JOB_TYPE="eval" \
  "${KDSD_PYTHON}" - "${eval_run}" "${summary}" <<'PY'
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

from scripts import evaluate_sd


class Log:
    @staticmethod
    def warning(*args, **kwargs):
        print("WARNING:", *args)


run_name = sys.argv[1]
summary_path = Path(sys.argv[2])
with summary_path.open("r", encoding="utf-8") as fh:
    summary = json.load(fh)

draft = summary.get("draft")
checkpoint_meta_path, checkpoint_meta = evaluate_sd._checkpoint_metadata_from_draft(draft)
cfg = OmegaConf.create(
    {
        "run_name": run_name,
        "draft": draft,
        "wandb": {
            "project": os.environ.get("WANDB_PROJECT", "cs552-kdsd"),
            "entity": os.environ.get("WANDB_ENTITY", ""),
            "dir": os.environ.get("WANDB_DIR", "wandb"),
            "mode": os.environ.get("WANDB_MODE", "online"),
            "resume": "allow",
        },
    }
)
evaluate_sd._report_eval_to_wandb(
    cfg=cfg,
    summary=summary,
    out_dir=summary_path.parent,
    checkpoint_meta=checkpoint_meta,
    checkpoint_meta_path=checkpoint_meta_path,
    log=Log(),
)
PY
}

run_eval() {
  local eval_run="$1"
  local draft="$2"
  local summary

  summary="$(summary_path "${eval_run}")"

  if ! is_true "${FORCE_RERUN}" && [[ -f "${summary}" ]]; then
    echo ">>> Skipping ${eval_run}; cached summary exists at ${summary}"
    report_cached_eval_to_wandb "${eval_run}" "${summary}"
    eval_result_runs+=("${eval_run}")
    return
  fi

  echo ">>> Evaluating ${eval_run}"
  echo ">>> draft=${draft}"
  WANDB_GROUP="${WANDB_GROUP}" \
  WANDB_JOB_TYPE="eval" \
  "${KDSD_PYTHON}" scripts/evaluate_sd.py \
    model=qwen25 "data=${DATA}" "${target_override[@]}" \
    "draft=${draft}" \
    "pretrained_checkpoint_root=${PRETRAINED_CHECKPOINT_ROOT}" \
    "prompts.jsonl=${EVAL_PROMPTS_JSONL}" \
    "prompts.hf_dataset=null" \
    "prompts.limit=${EVAL_PROMPTS_LIMIT}" \
    "runtime.mode=${EVAL_MODE}" \
    "runtime.temperature=${EVAL_TEMPERATURE}" \
    "runtime.top_p=${EVAL_TOP_P}" \
    "runtime.gamma=${EVAL_GAMMA}" \
    "runtime.max_new_tokens=${EVAL_MAX_NEW_TOKENS}" \
    "eval.backend=${EVAL_BACKEND}" \
    "eval.n_warmup=${EVAL_WARMUP}" \
    "eval.n_repeats=${EVAL_REPEATS}" \
    "wandb.enabled=${EVAL_REPORT_TO_WANDB}" \
    "results_dir=${RESULTS_ROOT}/${eval_run}" \
    "hydra.run.dir=${HYDRA_ROOT}/${eval_run}" \
    "run_name=${eval_run}"
  eval_result_runs+=("${eval_run}")
}

export PYTORCH_CUDA_ALLOC_CONF
mkdir -p "${RESULTS_ROOT}" "${HYDRA_ROOT}" "${PRETRAINED_CHECKPOINT_ROOT}"

echo ">>> Qwen2.5 eval sweep: ${EXPERIMENT_NAME}"
echo ">>> workspace root: ${WORKSPACE_ROOT}"
echo ">>> W&B group: ${WANDB_GROUP}"
echo ">>> run name prefix: ${RUN_NAME_PREFIX}"
echo ">>> losses/checkpoints: ${LOSSES}"
echo ">>> checkpoint root: ${CHECKPOINT_ROOT}"
echo ">>> results root: ${RESULTS_ROOT}"
echo ">>> pretrained checkpoint root: ${PRETRAINED_CHECKPOINT_ROOT}"
echo ">>> eval pretrained baseline: ${EVAL_PRETRAINED_BASELINE}"
echo ">>> eval prompts: ${EVAL_PROMPTS_JSONL}"
echo ">>> eval prompts limit: ${EVAL_PROMPTS_LIMIT}"
echo ">>> eval gamma/max_new: ${EVAL_GAMMA}/${EVAL_MAX_NEW_TOKENS}"
echo ">>> eval warmup/repeats: ${EVAL_WARMUP}/${EVAL_REPEATS}"
echo ">>> eval backend: ${EVAL_BACKEND}"
echo ">>> eval mode/temp/top_p: ${EVAL_MODE}/${EVAL_TEMPERATURE}/${EVAL_TOP_P}"
echo ">>> eval report to W&B: ${EVAL_REPORT_TO_WANDB}"
echo ">>> report cached evals to W&B: ${EVAL_REPORT_CACHED_TO_WANDB}"
echo ">>> force rerun: ${FORCE_RERUN}"

eval_result_runs=()

if is_true "${EVAL_PRETRAINED_BASELINE}"; then
  baseline_eval_run="${RUN_NAME_PREFIX}_pretrain_${DATA}_seed${SEED}_eval_g${EVAL_GAMMA}_max${EVAL_MAX_NEW_TOKENS}"
  run_eval "${baseline_eval_run}" "${DRAFT_ID}"
fi

for loss in ${LOSSES}; do
  train_run="${RUN_NAME_PREFIX}_${loss}_${DATA}_seed${SEED}"
  model_dir="$(checkpoint_dir_for_loss "${loss}")"
  eval_run="${train_run}_eval_g${EVAL_GAMMA}_max${EVAL_MAX_NEW_TOKENS}"

  if ! checkpoint_done "${model_dir}"; then
    echo ">>> WARNING: skipping ${train_run}; no completed checkpoint at ${model_dir}" >&2
    continue
  fi

  run_eval "${eval_run}" "${model_dir}"
done

if [[ "${#eval_result_runs[@]}" -eq 0 ]]; then
  echo "ERROR: no eval runs were produced. Check LOSSES, checkpoint root, and EVAL_PRETRAINED_BASELINE." >&2
  exit 1
fi

echo ">>> Final cached SD evaluation summary"
"${KDSD_PYTHON}" - "${RESULTS_ROOT}" "${eval_result_runs[@]}" <<'PY'
import json
import math
import sys
from pathlib import Path

results_root = Path(sys.argv[1])
runs = sys.argv[2:]
headers = (
    "run",
    "speedup",
    "accept",
    "avg_acc",
    "tok/s",
    "sd_s",
    "vanilla_s",
    "target_calls",
    "draft_calls",
)

rows = []
for run in runs:
    path = results_root / run / "eval_summary.json"
    if not path.exists():
        rows.append((run, "missing", "", "", "", "", "", "", ""))
        continue
    with path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    engines = summary.get("engines") or {}
    engine = engines.get("hf") or engines.get("vllm") or {}

    def fmt(value, pattern):
        if value is None:
            return ""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return ""
        if math.isnan(value):
            return ""
        return pattern % value

    rows.append(
        (
            run,
            fmt(summary.get("speedup"), "%.3fx"),
            fmt(summary.get("acceptance_rate"), "%.3f"),
            fmt(summary.get("avg_accepted_tokens"), "%.2f"),
            fmt(summary.get("tokens_per_second"), "%.2f"),
            fmt(summary.get("sd_time_s"), "%.2f"),
            fmt(summary.get("vanilla_time_s"), "%.2f"),
            str(engine.get("target_calls", "")),
            str(engine.get("draft_calls", "")),
        )
    )

widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
print(line)
print("-" * len(line))
for row in rows:
    print("  ".join(str(x).ljust(w) for x, w in zip(row, widths)))
PY
