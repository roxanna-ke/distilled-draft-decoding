#!/usr/bin/env bash
# Run the Week 2/3 non-CE KD loss ablation end to end.
#
# This script fixes data, training budget, eval prompts, and runtime decoding
# parameters, then trains and evaluates FKL/RKL/JSD with pure KD alpha 1.0.
# By default it trains on original UltraChat 50k responses and evaluates on that
# same data config's held-out eval split, avoiding cross-size split overlap.
# Run it inside the RunAI pod from the repository root:
#
#   bash scripts/run_loss_size_ablation.sh
#
# Common overrides:
#   TRAIN_STEPS=500 bash scripts/run_loss_size_ablation.sh
#   EVAL_LIMIT=200 N_REPEATS=3 bash scripts/run_loss_size_ablation.sh
#   MAX_NEW_TOKENS=128 bash scripts/run_loss_size_ablation.sh
#   EVAL_DATA=ultrachat_50k bash scripts/run_loss_size_ablation.sh
#   EXPERIMENT_ROOT=/scratch/my-runs bash scripts/run_loss_size_ablation.sh
#   RUN_DIR=/scratch/cs552-loss-ablation/<existing-run-dir> bash scripts/run_loss_size_ablation.sh
#   DATA_LIMIT=2000 bash scripts/run_loss_size_ablation.sh
# Set FORCE_RERUN=true to ignore completed eval/checkpoint artifacts.
# By default W&B logging is disabled; set WANDB=true to write offline W&B files.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/env.sh"
echo ">>> Python: ${KDSD_PYTHON}"

DATA="${DATA:-ultrachat_50k}"
EVAL_DATA="${EVAL_DATA:-${DATA}}"
DATA_LIMIT="${DATA_LIMIT:-}"
TRAIN_STEPS="${TRAIN_STEPS:-8000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
DATA_MAX_SEQ_LEN="${DATA_MAX_SEQ_LEN:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
JSD_TRAIN_BATCH_SIZE="${JSD_TRAIN_BATCH_SIZE:-2}"
JSD_GRAD_ACCUM_STEPS="${JSD_GRAD_ACCUM_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
KD_ALPHAS="${KD_ALPHAS:-1.0}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"

EVAL_LIMIT="${EVAL_LIMIT:-50}"
GAMMA_SWEEP="${GAMMA_SWEEP:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
RUNTIME_MODE="${RUNTIME_MODE:-sampling}"
RUNTIME_TEMPERATURE="${RUNTIME_TEMPERATURE:-1.0}"
RUNTIME_TOP_P="${RUNTIME_TOP_P:-0.9}"
N_WARMUP="${N_WARMUP:-1}"
N_REPEATS="${N_REPEATS:-3}"
EVAL_BACKEND="${EVAL_BACKEND:-manual}"

WANDB="${WANDB:-false}"
RUN_TAG="${RUN_TAG:-ultra50k_bugfix_s${TRAIN_STEPS}_seq${DATA_MAX_SEQ_LEN}_effbs$((TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS))}"
EVAL_TAG="${EVAL_TAG:-eval${EVAL_LIMIT}_m${MAX_NEW_TOKENS}}"
EVAL_JSONL="${EVAL_JSONL:-/scratch/cs552-data/processed/${EVAL_DATA}/eval.jsonl}"

TIMESTAMP="${TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/scratch/cs552-loss-ablation}"
RUN_DIR="${RUN_DIR:-${EXPERIMENT_ROOT}/${RUN_TAG}_${EVAL_TAG}_${TIMESTAMP}}"
RESULTS_ROOT="${RESULTS_ROOT:-${RUN_DIR}/results}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${RUN_DIR}/checkpoints}"
HYDRA_ROOT="${HYDRA_ROOT:-${RUN_DIR}/hydra}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
WANDB_DIR="${WANDB_DIR:-${RUN_DIR}/wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-cs552-kdsd}"
WANDB_MODE="${WANDB_MODE:-offline}"
FORCE_RERUN="${FORCE_RERUN:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RESULTS_ROOT}" "${CHECKPOINT_ROOT}" "${HYDRA_ROOT}" "${LOG_DIR}" "${WANDB_DIR}"
exec > >(tee -a "${LOG_DIR}/run.log") 2>&1

export WANDB_DIR WANDB_PROJECT WANDB_MODE

if [[ "${WANDB}" == "true" ]]; then
  REPORT_TO_WANDB=true
else
  REPORT_TO_WANDB=false
fi

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

is_true() {
  [[ "$1" == "true" || "$1" == "1" || "$1" == "yes" || "$1" == "y" ]]
}

eval_done() {
  [[ -f "${RESULTS_ROOT}/$1/eval_summary.json" ]]
}

checkpoint_done() {
  local model_dir="${CHECKPOINT_ROOT}/$1/model"
  [[ -f "${model_dir}/config.json" ]] && {
    compgen -G "${model_dir}/*.safetensors" >/dev/null || \
    compgen -G "${model_dir}/pytorch_model*.bin" >/dev/null
  }
}

run_eval() {
  local run_name="$1"
  local draft="$2"
  local gamma="$3"

  if ! is_true "${FORCE_RERUN}" && eval_done "${run_name}"; then
    log "Skipping eval ${run_name}; found ${RESULTS_ROOT}/${run_name}/eval_summary.json"
    return
  fi

  log "Evaluating ${run_name} (draft=${draft}, gamma=${gamma})"
  "${KDSD_PYTHON}" scripts/evaluate_sd.py \
    run_name="${run_name}" \
    results_dir="${RESULTS_ROOT}/${run_name}" \
    draft="${draft}" \
    prompts.jsonl="${EVAL_JSONL}" \
    prompts.hf_dataset=null \
    prompts.limit="${EVAL_LIMIT}" \
    runtime.mode="${RUNTIME_MODE}" \
    runtime.temperature="${RUNTIME_TEMPERATURE}" \
    runtime.top_p="${RUNTIME_TOP_P}" \
    runtime.gamma="${gamma}" \
    runtime.max_new_tokens="${MAX_NEW_TOKENS}" \
    eval.backend="${EVAL_BACKEND}" \
    eval.n_warmup="${N_WARMUP}" \
    eval.n_repeats="${N_REPEATS}" \
    hydra.run.dir="${HYDRA_ROOT}/${run_name}"
}

data_limit_args() {
  if [[ -n "${DATA_LIMIT}" ]]; then
    printf '+data.limit=%s\n' "${DATA_LIMIT}"
  fi
}

alpha_tag() {
  local alpha="$1"
  if [[ "${alpha}" == "0.5" ]]; then
    printf "a05"
  elif [[ "${alpha}" == "1.0" || "${alpha}" == "1" ]]; then
    printf "a1"
  else
    printf "a%s" "${alpha//./}"
  fi
}

train_loss() {
  local loss="$1"
  local run_name="$2"
  local alpha="$3"
  local batch_size="${TRAIN_BATCH_SIZE}"
  local grad_accum="${GRAD_ACCUM_STEPS}"
  local train_log="${LOG_DIR}/train_${run_name}.log"

  if [[ "${loss}" == "jsd" ]]; then
    batch_size="${JSD_TRAIN_BATCH_SIZE}"
    grad_accum="${JSD_GRAD_ACCUM_STEPS}"
  fi

  if ! is_true "${FORCE_RERUN}" && checkpoint_done "${run_name}"; then
    log "Skipping training ${run_name}; found completed checkpoint under ${CHECKPOINT_ROOT}/${run_name}/model"
    return
  fi

  log "Training ${run_name} (loss=${loss}, alpha=${alpha}, batch=${batch_size}, grad_accum=${grad_accum}); log=${train_log}"
  WANDB_NAME="${run_name}" "${KDSD_PYTHON}" scripts/train_size.py \
    run_name="${run_name}" \
    output_dir="${CHECKPOINT_ROOT}/${run_name}" \
    loss="${loss}" \
    data="${DATA}" \
    $(data_limit_args) \
    data.max_seq_len="${DATA_MAX_SEQ_LEN}" \
    train.max_steps="${TRAIN_STEPS}" \
    train.save_steps="${SAVE_STEPS}" \
    train.eval_steps="${SAVE_STEPS}" \
    train.save_total_limit="${SAVE_TOTAL_LIMIT}" \
    train.load_best_model_at_end=true \
    train.metric_for_best_model=eval_loss \
    train.greater_is_better=false \
    train.save_best_model=true \
    train.learning_rate="${LEARNING_RATE}" \
    train.lr_scheduler_type="${LR_SCHEDULER_TYPE}" \
    train.warmup_ratio="${WARMUP_RATIO}" \
    train.per_device_train_batch_size="${batch_size}" \
    train.gradient_accumulation_steps="${grad_accum}" \
    loss.alpha="${alpha}" \
    loss.temperature="${KD_TEMPERATURE}" \
    train.report_to_wandb="${REPORT_TO_WANDB}" \
    hydra.run.dir="${HYDRA_ROOT}/${run_name}" \
    2>&1 | tee -a "${train_log}"
}

log "Loss ablation configuration"
cat <<EOF
DATA=${DATA}
EVAL_DATA=${EVAL_DATA}
DATA_LIMIT=${DATA_LIMIT:-none}
TRAIN_STEPS=${TRAIN_STEPS}
SAVE_STEPS=${SAVE_STEPS}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}
DATA_MAX_SEQ_LEN=${DATA_MAX_SEQ_LEN}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}
JSD_TRAIN_BATCH_SIZE=${JSD_TRAIN_BATCH_SIZE}
JSD_GRAD_ACCUM_STEPS=${JSD_GRAD_ACCUM_STEPS}
LEARNING_RATE=${LEARNING_RATE}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE}
WARMUP_RATIO=${WARMUP_RATIO}
KD_ALPHAS=${KD_ALPHAS}
KD_TEMPERATURE=${KD_TEMPERATURE}
EVAL_JSONL=${EVAL_JSONL}
EVAL_LIMIT=${EVAL_LIMIT}
GAMMA_SWEEP=${GAMMA_SWEEP}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS}
RUNTIME_MODE=${RUNTIME_MODE}
RUNTIME_TEMPERATURE=${RUNTIME_TEMPERATURE}
RUNTIME_TOP_P=${RUNTIME_TOP_P}
N_WARMUP=${N_WARMUP}
N_REPEATS=${N_REPEATS}
EVAL_BACKEND=${EVAL_BACKEND}
WANDB=${WANDB}
WANDB_MODE=${WANDB_MODE}
WANDB_PROJECT=${WANDB_PROJECT}
FORCE_RERUN=${FORCE_RERUN}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}
RUN_DIR=${RUN_DIR}
RESULTS_ROOT=${RESULTS_ROOT}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT}
HYDRA_ROOT=${HYDRA_ROOT}
LOG_DIR=${LOG_DIR}
WANDB_DIR=${WANDB_DIR}
EOF

if [[ "${WANDB}" == "true" && "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  log "WANDB_API_KEY is not set; assuming 'wandb login' has already been run in this environment."
elif [[ "${WANDB}" == "true" && "${WANDB_MODE}" == "offline" ]]; then
  log "W&B is enabled in offline mode; no login is required. Local files will be saved in ${WANDB_DIR}."
fi

log "Preparing fixed eval data split"
"${KDSD_PYTHON}" scripts/prepare_data.py data="${EVAL_DATA}"

if [[ ! -f "${EVAL_JSONL}" ]]; then
  echo "ERROR: expected eval JSONL not found: ${EVAL_JSONL}" >&2
  exit 1
fi
log "Final SD eval prompts will use ${EVAL_JSONL}"

SUMMARY_RUNS=()

for loss in fkl rkl jsd; do
  for alpha in ${KD_ALPHAS}; do
    atag="$(alpha_tag "${alpha}")"
    run_name="${loss}_${RUN_TAG}_${atag}"
    if [[ "${loss}" == "jsd" ]]; then
      run_name="${run_name}_effbs$((JSD_TRAIN_BATCH_SIZE * JSD_GRAD_ACCUM_STEPS))"
    else
      run_name="${run_name}_effbs$((TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS))"
    fi
    train_loss "${loss}" "${run_name}" "${alpha}"
    for gamma in ${GAMMA_SWEEP}; do
      eval_name="eval_${run_name}_g${gamma}_${EVAL_TAG}"
      run_eval "${eval_name}" "${CHECKPOINT_ROOT}/${run_name}/model" "${gamma}"
      SUMMARY_RUNS+=("${eval_name}")
    done
  done
done

log "Loss ablation complete"
cat <<EOF
All outputs for this ablation were written under:
  ${RUN_DIR}

Eval results:
$(for run_name in "${SUMMARY_RUNS[@]}"; do printf '  %s/%s\n' "${RESULTS_ROOT}" "${run_name}"; done)

Checkpoints:
  ${CHECKPOINT_ROOT}
  ${CHECKPOINT_ROOT}/<run_name>/best_model

Logs:
  ${LOG_DIR}/run.log
  ${LOG_DIR}/train_<run_name>.log

W&B local files:
  ${WANDB_DIR}

Compare each eval_summary.json on acceptance_rate, avg_accepted_tokens,
speedup, tokens_per_second, target_calls, draft_calls, draft_forward_s,
and target_forward_s.
EOF
