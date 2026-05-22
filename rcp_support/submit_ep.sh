#!/bin/bash
# CS-552 Run:AI launcher for A100 bf16 KD training.
#
# Submit from your laptop:
#   GASPAR=ke GROUP=g67 ./rcp_support/submit_ep.sh losses10k
#
# Useful overrides:
#   LOSSES=fkl,rkl,jsd DATASET=ultrachat_50k MAX_STEPS=8000 ./rcp_support/submit_ep.sh losses50k
#   WANDB_MODE=offline ./rcp_support/submit_ep.sh debug

set -euo pipefail

# ============== EDIT / OVERRIDE THESE ==============
GASPAR="${GASPAR:-ke}"              # EPFL GASPAR username.
GROUP="${GROUP:-g67}"               # Team PVC id, e.g. g07.
WANDB_MODE="${WANDB_MODE:-online}"  # online, offline, or disabled.

REPO_URL="${REPO_URL:-https://github.com/roxanna-ke/distilled-draft-decoding.git}"  # Used only when SYNC_REPO=true.
REPO_BRANCH="${REPO_BRANCH:-train}" # Used only when SYNC_REPO=true.
SYNC_REPO="${SYNC_REPO:-true}"     # true: fetch/reset remote; false: use existing REPO_DIR as-is.
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/scratch/cs552-mnlp-kzy}"
REPO_DIR="${REPO_DIR:-${WORKSPACE_ROOT}/repos/distilled-draft-decoding-train}"

LOSSES="${LOSSES:-${LOSS:-fkl,rkl,jsd}}"
LOSS_LIST="${LOSSES//,/ }"
DATASET="${DATASET:-ultrachat_50k}"
MAX_STEPS="${MAX_STEPS:-8000}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
KD_ALPHA="${KD_ALPHA:-1.0}"
KD_TEMPERATURE="${KD_TEMPERATURE:-2.0}"
SEED="${SEED:-42}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-ultra50k_s8000_seq512_a1_temp2}"

HF_HOME_DIR="${HF_HOME_DIR:-${WORKSPACE_ROOT}/hf_cache}"
RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_ROOT}/results}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${WORKSPACE_ROOT}/checkpoints}"
DATA_DIR="${DATA_DIR:-${WORKSPACE_ROOT}/data}"
HYDRA_OUTPUTS_DIR="${HYDRA_OUTPUTS_DIR:-${WORKSPACE_ROOT}/hydra}"
WANDB_DIR="${WANDB_DIR:-${WORKSPACE_ROOT}/wandb}"
LINK_ARTIFACT_DIRS="${LINK_ARTIFACT_DIRS:-false}"  # true: create repo-local symlinks to artifact dirs.

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

N_SAMPLES="${N_SAMPLES:-50000}"
VAL_SAMPLES="${VAL_SAMPLES:-512}"
EVAL_SAMPLES="${EVAL_SAMPLES:-256}"
HF_DATASET_NAME="${HF_DATASET_NAME:-HuggingFaceH4/ultrachat_200k}"
HF_DATASET_SPLIT="${HF_DATASET_SPLIT:-train_sft}"

LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
OVERFIT_SAMPLES="${OVERFIT_SAMPLES:-0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-null}"
# ===================================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
  echo "ERROR: set GASPAR to your EPFL username." >&2
  exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
  echo "ERROR: set GROUP to your team number, e.g. g07." >&2
  exit 1
fi

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_MODE=online requires WANDB_API_KEY. Use WANDB_MODE=offline otherwise." >&2
  exit 1
fi

GPUS=1
NODE="${NODE:-a100-40g}"
SUFFIX="${1:-train-temp2}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"
IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

read -r -d '' POD_COMMAND <<'POD' || true
set -euo pipefail

for required_name in SYNC_REPO REPO_DIR HF_HOME_DIR RESULTS_DIR WANDB_DIR CHECKPOINTS_DIR DATA_DIR HYDRA_OUTPUTS_DIR LINK_ARTIFACT_DIRS; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "ERROR: ${required_name} is empty inside the pod." >&2
    exit 1
  fi
done

mkdir -p \
  "${HF_HOME_DIR}" \
  "${RESULTS_DIR}" \
  "${WANDB_DIR}" \
  "${CHECKPOINTS_DIR}" \
  "${DATA_DIR}" \
  "${HYDRA_OUTPUTS_DIR}" \
  "$(dirname "${REPO_DIR}")"

export HF_HOME="${HF_HOME_DIR}"
export HF_HUB_CACHE="${HF_HOME_DIR}/hub"
export HF_DATASETS_CACHE="${HF_HOME_DIR}/datasets"
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

if command -v python3 >/dev/null 2>&1; then
  ln -sf "$(command -v python3)" /usr/local/bin/python 2>/dev/null || true
fi

lock_file="${REPO_DIR}.lock"
exec 9>"${lock_file}"
if command -v flock >/dev/null 2>&1; then
  echo ">>> Waiting for checkout lock: ${lock_file}"
  flock 9
fi

configure_git_safe_directory() {
  export GIT_CONFIG_COUNT="${GIT_CONFIG_COUNT:-0}"
  local idx="${GIT_CONFIG_COUNT}"
  export "GIT_CONFIG_KEY_${idx}=safe.directory"
  export "GIT_CONFIG_VALUE_${idx}=${REPO_DIR}"
  export GIT_CONFIG_COUNT="$((idx + 1))"
  git config --global --add safe.directory "${REPO_DIR}" 2>/dev/null || true
}

configure_git_safe_directory

if [[ "${SYNC_REPO}" == "true" ]]; then
  for required_name in REPO_URL REPO_BRANCH; do
    if [[ -z "${!required_name:-}" ]]; then
      echo "ERROR: ${required_name} is empty inside the pod." >&2
      exit 1
    fi
  done

  if ! git ls-remote --exit-code --heads "${REPO_URL}" "${REPO_BRANCH}" >/dev/null; then
    echo "ERROR: cannot read branch ${REPO_BRANCH} from ${REPO_URL}" >&2
    exit 1
  fi

  if [[ ! -d "${REPO_DIR}/.git" ]]; then
    if [[ -e "${REPO_DIR}" ]]; then
      echo "ERROR: ${REPO_DIR} exists but is not a git checkout." >&2
      exit 1
    fi
    echo ">>> Cloning ${REPO_URL} into ${REPO_DIR}"
    git clone --no-checkout --origin origin "${REPO_URL}" "${REPO_DIR}"
  fi

  git -C "${REPO_DIR}" remote set-url origin "${REPO_URL}"
  cd "${REPO_DIR}"
  echo ">>> Fetching origin/${REPO_BRANCH}"
  git fetch --prune origin "+refs/heads/${REPO_BRANCH}:refs/remotes/origin/${REPO_BRANCH}"
  git checkout -B "${REPO_BRANCH}" "origin/${REPO_BRANCH}"
  git reset --hard "origin/${REPO_BRANCH}"
  git clean -ffd
elif [[ "${SYNC_REPO}" == "false" ]]; then
  if [[ ! -d "${REPO_DIR}" ]]; then
    echo "ERROR: SYNC_REPO=false requires an existing repo directory at ${REPO_DIR}" >&2
    exit 1
  fi
  if [[ ! -f "${REPO_DIR}/scripts/train.py" ]]; then
    echo "ERROR: ${REPO_DIR} does not look like the training repo; missing scripts/train.py" >&2
    exit 1
  fi
  cd "${REPO_DIR}"
  echo ">>> Using existing repo directory at ${REPO_DIR}"
else
  echo "ERROR: SYNC_REPO must be true or false, got ${SYNC_REPO}" >&2
  exit 1
fi

chmod +x rcp_support/train_ep_pod.sh 2>/dev/null || true
exec bash rcp_support/train_ep_pod.sh
POD

POD_COMMAND_B64="$(printf '%s' "${POD_COMMAND}" | base64 | tr -d '\n')"

echo ">>> Submitting A100 bf16 training job ${JOB_NAME}"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu "${GPUS}" \
  --large-shm \
  --node-pools "${NODE}" \
  --working-dir /scratch \
  --environment WORKSPACE_ROOT="${WORKSPACE_ROOT}" \
  --environment HF_HOME="${HF_HOME_DIR}" \
  --environment HF_HOME_DIR="${HF_HOME_DIR}" \
  --environment HF_TOKEN="${HF_TOKEN:-}" \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment TOKENIZERS_PARALLELISM=false \
  --environment PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --environment WANDB_DIR="${WANDB_DIR}" \
  --environment WANDB_MODE="${WANDB_MODE}" \
  --environment WANDB_PROJECT="${WANDB_PROJECT:-cs552-kdsd}" \
  --environment WANDB_NAME="${JOB_NAME}" \
  --environment WANDB_ENTITY="${WANDB_ENTITY:-}" \
  --environment WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --environment REPO_URL="${REPO_URL}" \
  --environment REPO_BRANCH="${REPO_BRANCH}" \
  --environment SYNC_REPO="${SYNC_REPO}" \
  --environment REPO_DIR="${REPO_DIR}" \
  --environment LOSSES="${LOSSES}" \
  --environment DATASET="${DATASET}" \
  --environment MAX_STEPS="${MAX_STEPS}" \
  --environment MAX_SEQ_LEN="${MAX_SEQ_LEN}" \
  --environment PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --environment PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --environment GRAD_ACCUM="${GRAD_ACCUM}" \
  --environment LEARNING_RATE="${LEARNING_RATE}" \
  --environment WEIGHT_DECAY="${WEIGHT_DECAY}" \
  --environment WARMUP_RATIO="${WARMUP_RATIO}" \
  --environment LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE}" \
  --environment KD_ALPHA="${KD_ALPHA}" \
  --environment KD_TEMPERATURE="${KD_TEMPERATURE}" \
  --environment SEED="${SEED}" \
  --environment RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX}" \
  --environment TARGET_MODEL="${TARGET_MODEL}" \
  --environment DRAFT_MODEL="${DRAFT_MODEL}" \
  --environment MODEL_DTYPE="${MODEL_DTYPE}" \
  --environment ATTN_IMPL="${ATTN_IMPL}" \
  --environment N_SAMPLES="${N_SAMPLES}" \
  --environment VAL_SAMPLES="${VAL_SAMPLES}" \
  --environment EVAL_SAMPLES="${EVAL_SAMPLES}" \
  --environment HF_DATASET_NAME="${HF_DATASET_NAME}" \
  --environment HF_DATASET_SPLIT="${HF_DATASET_SPLIT}" \
  --environment LOGGING_STEPS="${LOGGING_STEPS}" \
  --environment SAVE_STEPS="${SAVE_STEPS}" \
  --environment EVAL_STEPS="${EVAL_STEPS}" \
  --environment SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
  --environment DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}" \
  --environment OVERFIT_SAMPLES="${OVERFIT_SAMPLES}" \
  --environment RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT}" \
  --environment RESULTS_DIR="${RESULTS_DIR}" \
  --environment CHECKPOINTS_DIR="${CHECKPOINTS_DIR}" \
  --environment DATA_DIR="${DATA_DIR}" \
  --environment HYDRA_OUTPUTS_DIR="${HYDRA_OUTPUTS_DIR}" \
  --environment LINK_ARTIFACT_DIRS="${LINK_ARTIFACT_DIRS}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "set -euo pipefail; printf '%s' '${POD_COMMAND_B64}' | base64 -d | /bin/bash"

cat <<EOF

>>> Training job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
List jobs:         runai list jobs -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

Training entrypoint:
rcp_support/train_ep_pod.sh
EOF
