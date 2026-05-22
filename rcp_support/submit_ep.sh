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

REPO_URL="${REPO_URL:-https://github.com/CS-552/open-project-m2-shallowseek.git}"  # Used only when SYNC_REPO=true.
REPO_BRANCH="${REPO_BRANCH:-train}" # Used only when SYNC_REPO=true.
SYNC_REPO="${SYNC_REPO:-false}"     # true: fetch/reset remote; false: use existing REPO_DIR as-is.
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/scratch/cs552-mnlp-kzy}"
REPO_DIR="${REPO_DIR:-${WORKSPACE_ROOT}/repos/open-project-m2-shallowseek}"

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

if [[ -z "${RUN_COMMAND:-}" ]]; then
  read -r -d '' RUN_COMMAND <<EOF || true
for loss_name in ${LOSS_LIST}; do
  run_name="\${loss_name}_${RUN_NAME_SUFFIX}"
  echo ">>> Starting training: loss=\${loss_name} run_name=\${run_name}"
  python scripts/train.py \\
    loss="\${loss_name}" data="${DATASET}" seed="${SEED}" run_name="\${run_name}" \\
    output_dir="${CHECKPOINTS_DIR}/\${run_name}" \\
    results_dir="${RESULTS_DIR}/\${run_name}" \\
    data_root="${DATA_DIR}" \\
    hf_cache="${HF_HOME_DIR}" \\
    hydra.run.dir="${HYDRA_OUTPUTS_DIR}/\${run_name}" \\
    model.target="${TARGET_MODEL}" model.draft_default="${DRAFT_MODEL}" \\
    model.device=cuda model.dtype="${MODEL_DTYPE}" model.attn_impl="${ATTN_IMPL}" \\
    model.trust_remote_code=false \\
    data.response_source=original data.n_samples="${N_SAMPLES}" \\
    data.val_samples="${VAL_SAMPLES}" data.eval_samples="${EVAL_SAMPLES}" \\
    data.max_seq_len="${MAX_SEQ_LEN}" \\
    data.hf_dataset.name="${HF_DATASET_NAME}" data.hf_dataset.split="${HF_DATASET_SPLIT}" \\
    loss.alpha="${KD_ALPHA}" loss.temperature="${KD_TEMPERATURE}" \\
    train.max_steps="${MAX_STEPS}" train.num_train_epochs=1 \\
    train.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \\
    train.per_device_eval_batch_size="${PER_DEVICE_EVAL_BATCH_SIZE}" \\
    train.gradient_accumulation_steps="${GRAD_ACCUM}" \\
    train.learning_rate="${LEARNING_RATE}" train.weight_decay="${WEIGHT_DECAY}" \\
    train.warmup_ratio="${WARMUP_RATIO}" train.lr_scheduler_type="${LR_SCHEDULER_TYPE}" \\
    train.logging_steps="${LOGGING_STEPS}" train.save_steps="${SAVE_STEPS}" \\
    train.eval_steps="${EVAL_STEPS}" train.save_total_limit="${SAVE_TOTAL_LIMIT}" \\
    ++train.load_best_model_at_end=true ++train.metric_for_best_model=eval_loss \\
    ++train.greater_is_better=false ++train.save_best_model=true \\
    train.bf16=true train.fp16=false train.gradient_checkpointing=true \\
    train.dataloader_drop_last=true train.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \\
    train.remove_unused_columns=false train.report_to_wandb=true \\
    train.resume_from_checkpoint="${RESUME_FROM_CHECKPOINT}" \\
    train.overfit_samples="${OVERFIT_SAMPLES}" train.compile_target=false \\
    eval.n_warmup=1 eval.n_repeats=3 eval.run_vanilla_baseline=true eval.write_generations=true \\
    runtime.mode=sampling runtime.temperature=1.0 runtime.top_p=0.9 \\
    runtime.gamma=4 runtime.max_new_tokens=256 \\
    wandb.enabled=true wandb.project="${WANDB_PROJECT:-cs552-kdsd}" \\
    wandb.dir="${WANDB_DIR}" wandb.mode="${WANDB_MODE}"
  echo ">>> Finished training: loss=\${loss_name} run_name=\${run_name}"
done
EOF
fi

RUN_COMMAND_B64="$(printf '%s' "${RUN_COMMAND}" | base64 | tr -d '\n')"

GPUS=1
NODE="${NODE:-a100-40g}"
SUFFIX="${1:-train-temp2}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"
IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

read -r -d '' BOOTSTRAP_COMMAND <<'BOOTSTRAP' || true
set -euo pipefail

for required_name in SYNC_REPO REPO_DIR HF_HOME_DIR RESULTS_DIR WANDB_DIR CHECKPOINTS_DIR DATA_DIR HYDRA_OUTPUTS_DIR LINK_ARTIFACT_DIRS RUN_COMMAND_B64; do
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

if [[ "${LINK_ARTIFACT_DIRS}" == "true" ]]; then
  link_artifact_dir checkpoints "${CHECKPOINTS_DIR}"
  link_artifact_dir data "${DATA_DIR}"
  link_artifact_dir outputs "${HYDRA_OUTPUTS_DIR}"
  link_artifact_dir wandb "${WANDB_DIR}"
elif [[ "${LINK_ARTIFACT_DIRS}" != "false" ]]; then
  echo "ERROR: LINK_ARTIFACT_DIRS must be true or false, got ${LINK_ARTIFACT_DIRS}" >&2
  exit 1
fi

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

if [[ -d .git ]] && command -v git >/dev/null 2>&1; then
  echo ">>> Repo commit: $(git rev-parse HEAD)"
else
  echo ">>> Repo source: existing directory without git metadata"
fi
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
  --environment WANDB_NAME="${JOB_NAME}" \
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
  --environment LINK_ARTIFACT_DIRS="${LINK_ARTIFACT_DIRS}" \
  --environment RUN_COMMAND_B64="${RUN_COMMAND_B64}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "set -euo pipefail; printf '%s' '${BOOTSTRAP_B64}' | base64 -d | /bin/bash"

cat <<EOF

>>> Training job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
List jobs:         runai list jobs -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

Default command:
${RUN_COMMAND}
EOF
