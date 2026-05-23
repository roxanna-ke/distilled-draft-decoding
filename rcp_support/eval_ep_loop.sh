#!/bin/bash
# Run SD evaluation with W&B for one or more draft models.
#
# Run inside the Run:AI pod after checkpoints are available, for example:
#   cd /scratch/cs552-mnlp-kzy/repos/distilled-draft-decoding-train
#   WANDB_MODE=online bash rcp_support/eval_ep_loop.sh

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/scratch/cs552-mnlp-kzy}"
REPO_DIR="${REPO_DIR:-${WORKSPACE_ROOT}/repos/distilled-draft-decoding-train}"
HF_HOME_DIR="${HF_HOME_DIR:-${WORKSPACE_ROOT}/hf_cache}"
RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_ROOT}/results}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${WORKSPACE_ROOT}/checkpoints}"
DATA_DIR="${DATA_DIR:-${WORKSPACE_ROOT}/data}"
HYDRA_OUTPUTS_DIR="${HYDRA_OUTPUTS_DIR:-${WORKSPACE_ROOT}/hydra}"
WANDB_DIR="${WANDB_DIR:-${WORKSPACE_ROOT}/wandb}"

DATASET="${DATASET:-ultrachat_50k}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-ultra50k_s8000_seq512_a1_temp2}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
SEED="${SEED:-42}"

EVAL_DRAFTS="${EVAL_DRAFTS:-fkl}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-${DATA_DIR}/processed/${DATASET}/eval.jsonl}"
EVAL_PROMPT_LIMIT="${EVAL_PROMPT_LIMIT:-50}"
EVAL_N_WARMUP="${EVAL_N_WARMUP:-1}"
EVAL_N_REPEATS="${EVAL_N_REPEATS:-3}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_GAMMA="${EVAL_GAMMA:-4}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.9}"
EVAL_RUN_VANILLA_BASELINE="${EVAL_RUN_VANILLA_BASELINE:-true}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-cs552-kdsd}"

mkdir -p "${HF_HOME_DIR}" "${RESULTS_DIR}" "${HYDRA_OUTPUTS_DIR}" "${WANDB_DIR}"
export HF_HOME="${HF_HOME_DIR}"
export HF_HUB_CACHE="${HF_HOME_DIR}/hub"
export HF_DATASETS_CACHE="${HF_HOME_DIR}/datasets"
export WANDB_DIR="${WANDB_DIR}"
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

cd "${REPO_DIR}"

draft_spec_for_name() {
  local name="$1"
  case "${name}" in
    pretrained)
      printf '%s\n' "${DRAFT_MODEL}"
      ;;
    fkl|rkl|jsd)
      printf '%s\n' "${CHECKPOINTS_DIR}/${name}_${RUN_NAME_SUFFIX}/model"
      ;;
    *)
      echo "ERROR: unknown eval draft '${name}'. Expected one of: pretrained,fkl,rkl,jsd" >&2
      return 1
      ;;
  esac
}

IFS=',' read -r -a draft_names <<< "${EVAL_DRAFTS}"

for draft_name in "${draft_names[@]}"; do
  draft_name="$(printf '%s' "${draft_name}" | xargs)"
  [[ -n "${draft_name}" ]] || continue

  draft_spec="$(draft_spec_for_name "${draft_name}")"
  if [[ "${draft_name}" != "pretrained" && ! -d "${draft_spec}" ]]; then
    echo "ERROR: missing checkpoint model directory for ${draft_name}: ${draft_spec}" >&2
    exit 1
  fi

  eval_run_name="eval_${draft_name}_${RUN_NAME_SUFFIX}"
  eval_results_dir="${RESULTS_DIR}/${eval_run_name}"
  hydra_dir="${HYDRA_OUTPUTS_DIR}/${eval_run_name}"

  cmd=(
    python scripts/evaluate_sd.py
    data="${DATASET}"
    seed="${SEED}"
    run_name="${eval_run_name}"
    draft="${draft_spec}"
    results_dir="${eval_results_dir}"
    hf_cache="${HF_HOME_DIR}"
    hydra.run.dir="${hydra_dir}"
    model.target="${TARGET_MODEL}"
    model.draft_default="${DRAFT_MODEL}"
    model.device=cuda
    model.dtype="${MODEL_DTYPE}"
    model.attn_impl="${ATTN_IMPL}"
    model.trust_remote_code=false
    prompts.limit="${EVAL_PROMPT_LIMIT}"
    eval.n_warmup="${EVAL_N_WARMUP}"
    eval.n_repeats="${EVAL_N_REPEATS}"
    eval.run_vanilla_baseline="${EVAL_RUN_VANILLA_BASELINE}"
    eval.write_generations=true
    runtime.mode=sampling
    runtime.temperature="${EVAL_TEMPERATURE}"
    runtime.top_p="${EVAL_TOP_P}"
    runtime.gamma="${EVAL_GAMMA}"
    runtime.max_new_tokens="${EVAL_MAX_NEW_TOKENS}"
    wandb.enabled=true
    wandb.project="${WANDB_PROJECT}"
    wandb.dir="${WANDB_DIR}"
    wandb.mode="${WANDB_MODE}"
  )

  if [[ -f "${EVAL_PROMPTS_JSONL}" ]]; then
    cmd+=(prompts.jsonl="${EVAL_PROMPTS_JSONL}")
  else
    echo ">>> Prompt file not found: ${EVAL_PROMPTS_JSONL}"
    echo ">>> Falling back to config default HF eval prompt source."
  fi

  echo ">>> Starting eval: draft=${draft_name} draft_spec=${draft_spec}"
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
  echo ">>> Finished eval: draft=${draft_name} results=${eval_results_dir}"
done
