#!/bin/bash
# CS-552 Run:AI launcher for A100 bf16 KD training.
#
# Submit from your laptop:
#   GASPAR=kzy GROUP=g67 ./rcp_support/submit_ep.sh fkl10k
#
# Useful overrides:
#   LOSS=jsd DATASET=ultrachat_10k TRAIN_STEPS=4000 ./rcp_support/submit_ep.sh jsd10k
#   WANDB_MODE=offline ./rcp_support/submit_ep.sh debug

set -euo pipefail

# ============== EDIT / OVERRIDE THESE ==============
GASPAR="${GASPAR:-kzy}"             # EPFL GASPAR username.
GROUP="${GROUP:-g67}"               # Team PVC id, e.g. g07.
WANDB_MODE="${WANDB_MODE:-offline}" # online, offline, or disabled.

REPO_URL="${REPO_URL:-https://github.com/roxanna-ke/distilled-draft-decoding.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
SYNC_REPO="${SYNC_REPO:-true}"      # true: fetch/reset remote; false: use existing REPO_DIR as-is.
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/scratch/cs552-mnlp-kzy}"
REPO_DIR="${REPO_DIR:-${WORKSPACE_ROOT}/repos/distilled-draft-decoding}"

LOSS="${LOSS:-fkl}"
DATASET="${DATASET:-ultrachat_10k}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
KD_ALPHA="${KD_ALPHA:-0.5}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-kd_${LOSS}_${DATASET}_targetgen_bf16_seed${SEED}}"

HF_HOME_DIR="${HF_HOME_DIR:-${WORKSPACE_ROOT}/hf_cache}"
RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_ROOT}/logs/results}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${WORKSPACE_ROOT}/checkpoints}"
DATA_DIR="${DATA_DIR:-${WORKSPACE_ROOT}/data}"
HYDRA_OUTPUTS_DIR="${HYDRA_OUTPUTS_DIR:-${WORKSPACE_ROOT}/hydra}"
WANDB_DIR="${WANDB_DIR:-${WORKSPACE_ROOT}/wandb}"
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

RUN_COMMAND="${RUN_COMMAND:-python scripts/train.py \
  loss=${LOSS} data=${DATASET} seed=${SEED} run_name=${RUN_NAME} \
  model.device=cuda model.dtype=bfloat16 \
  train.use_amp=true train.amp_dtype=bfloat16 \
  loss.alpha=${KD_ALPHA} loss.temperature=${KD_TEMPERATURE} \
  train.steps=${TRAIN_STEPS} train.max_length=${MAX_LENGTH} \
  train.batch_size=${BATCH_SIZE} train.gradient_accumulation_steps=${GRAD_ACCUM} \
  train.learning_rate=${LEARNING_RATE} \
  train.output_dir=${CHECKPOINTS_DIR}/${RUN_NAME} \
  train.wandb.enabled=true train.wandb.project=${WANDB_PROJECT:-cs552-kdsd} \
  train.wandb.mode=${WANDB_MODE}
}"

RUN_COMMAND_B64="$(printf '%s' "${RUN_COMMAND}" | base64 | tr -d '\n')"

GPUS=1
NODE="${NODE:-a100-40g}"
SUFFIX="${1:-train}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"
IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

read -r -d '' BOOTSTRAP_COMMAND <<'BOOTSTRAP' || true
set -euo pipefail

for required_name in REPO_URL REPO_BRANCH SYNC_REPO REPO_DIR HF_HOME_DIR RESULTS_DIR WANDB_DIR CHECKPOINTS_DIR DATA_DIR HYDRA_OUTPUTS_DIR RUN_COMMAND_B64; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "ERROR: ${required_name} is empty inside the pod." >&2
    exit 1
  fi
done

RUN_COMMAND="$(printf '%s' "${RUN_COMMAND_B64}" | base64 -d)"

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
  if [[ ! -d "${REPO_DIR}/.git" ]]; then
    echo "ERROR: SYNC_REPO=false requires an existing git checkout at ${REPO_DIR}" >&2
    exit 1
  fi
  cd "${REPO_DIR}"
  echo ">>> Using existing checkout at ${REPO_DIR}"
else
  echo "ERROR: SYNC_REPO must be true or false, got ${SYNC_REPO}" >&2
  exit 1
fi

link_artifact_dir() {
  local name="$1"
  local target="$2"
  if [[ -L "${name}" ]]; then
    rm "${name}"
  elif [[ -e "${name}" ]]; then
    echo "ERROR: ${REPO_DIR}/${name} exists and is not a symlink." >&2
    exit 1
  fi
  mkdir -p "${target}"
  ln -s "${target}" "${name}"
}

link_artifact_dir checkpoints "${CHECKPOINTS_DIR}"
link_artifact_dir data "${DATA_DIR}"
link_artifact_dir outputs "${HYDRA_OUTPUTS_DIR}"
link_artifact_dir wandb "${WANDB_DIR}"

echo ">>> CUDA sanity check"
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu=", torch.cuda.get_device_name(0))
    print("bf16_supported=", torch.cuda.is_bf16_supported())
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("bf16 is not supported by this GPU; request NODE=a100-40g")
PY

echo ">>> Checked out ${REPO_BRANCH} at $(git rev-parse HEAD)"
echo ">>> Personal workspace: $(dirname "${REPO_DIR}")"
echo ">>> Running command:"
printf '%s\n' "${RUN_COMMAND}"
eval "${RUN_COMMAND}"
BOOTSTRAP

BOOTSTRAP_B64="$(printf '%s' "${BOOTSTRAP_COMMAND}" | base64 | tr -d '\n')"

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
  --environment WANDB_NAME="${RUN_NAME}" \
  --environment WANDB_ENTITY="${WANDB_ENTITY:-}" \
  --environment WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --environment REPO_URL="${REPO_URL}" \
  --environment REPO_BRANCH="${REPO_BRANCH}" \
  --environment SYNC_REPO="${SYNC_REPO}" \
  --environment REPO_DIR="${REPO_DIR}" \
  --environment RESULTS_DIR="${RESULTS_DIR}" \
  --environment CHECKPOINTS_DIR="${CHECKPOINTS_DIR}" \
  --environment DATA_DIR="${DATA_DIR}" \
  --environment HYDRA_OUTPUTS_DIR="${HYDRA_OUTPUTS_DIR}" \
  --environment RUN_COMMAND_B64="${RUN_COMMAND_B64}" \
  --environment BOOTSTRAP_B64="${BOOTSTRAP_B64}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc 'set -euo pipefail; printf "%s" "${BOOTSTRAP_B64}" | base64 -d | /bin/bash'

cat <<EOF

>>> Training job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
List jobs:         runai list jobs -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

Default command:
${RUN_COMMAND}
EOF
