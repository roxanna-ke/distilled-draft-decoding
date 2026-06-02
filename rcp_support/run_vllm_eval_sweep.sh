#!/bin/bash
# Run inside the Run:AI pod after the repository checkout has been prepared.
# Expects the same directory conventions as rcp_support/eval_ep_loop.sh, but
# forces the vLLM eval backend.

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
SEED="${SEED:-42}"

EVAL_DRAFTS="${EVAL_DRAFTS:-pretrained,fkl,rkl,jsd}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-${DATA_DIR}/processed/${DATASET}/eval.jsonl}"
EVAL_PROMPT_LIMIT="${EVAL_PROMPT_LIMIT:-50}"
EVAL_N_WARMUP="${EVAL_N_WARMUP:-0}"
EVAL_N_REPEATS="${EVAL_N_REPEATS:-1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_GAMMA="${EVAL_GAMMA:-4}"
EVAL_MODE="${EVAL_MODE:-sampling}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.9}"
EVAL_RUN_VANILLA_BASELINE="${EVAL_RUN_VANILLA_BASELINE:-true}"

VLLM_REQUEST_BATCH_SIZE="${VLLM_REQUEST_BATCH_SIZE:-8}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
VLLM_DRAFT_TP_SIZE="${VLLM_DRAFT_TP_SIZE:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_SWAP_SPACE="${VLLM_SWAP_SPACE:-0}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"
VLLM_DISABLE_LOG_STATS="${VLLM_DISABLE_LOG_STATS:-false}"

mkdir -p "${HF_HOME_DIR}" "${RESULTS_DIR}" "${HYDRA_OUTPUTS_DIR}" "${WANDB_DIR}"
export HF_HOME="${HF_HOME_DIR}"
export HF_HUB_CACHE="${HF_HOME_DIR}/hub"
export HF_DATASETS_CACHE="${HF_HOME_DIR}/datasets"
export WANDB_DIR="${WANDB_DIR}"
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

cd "${REPO_DIR}"

echo ">>> Repo branch: ${REPO_BRANCH:-unknown}"
echo ">>> Checkpoints dir: ${CHECKPOINTS_DIR}"
echo ">>> Results root: ${RESULTS_DIR}"
echo ">>> Draft set: ${EVAL_DRAFTS}"

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
    model.trust_remote_code=false
    prompts.limit="${EVAL_PROMPT_LIMIT}"
    eval.backend=vllm
    eval.n_warmup="${EVAL_N_WARMUP}"
    eval.n_repeats="${EVAL_N_REPEATS}"
    eval.run_vanilla_baseline="${EVAL_RUN_VANILLA_BASELINE}"
    eval.write_generations=true
    eval.vllm.request_batch_size="${VLLM_REQUEST_BATCH_SIZE}"
    eval.vllm.tensor_parallel_size="${VLLM_TP_SIZE}"
    eval.vllm.draft_tensor_parallel_size="${VLLM_DRAFT_TP_SIZE}"
    eval.vllm.max_model_len="${VLLM_MAX_MODEL_LEN}"
    eval.vllm.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION}"
    eval.vllm.swap_space="${VLLM_SWAP_SPACE}"
    eval.vllm.enforce_eager="${VLLM_ENFORCE_EAGER}"
    eval.vllm.disable_log_stats="${VLLM_DISABLE_LOG_STATS}"
    runtime.mode="${EVAL_MODE}"
    runtime.temperature="${EVAL_TEMPERATURE}"
    runtime.top_p="${EVAL_TOP_P}"
    runtime.gamma="${EVAL_GAMMA}"
    runtime.max_new_tokens="${EVAL_MAX_NEW_TOKENS}"
    wandb.enabled=false
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
