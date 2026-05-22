#!/bin/bash
# CS-552 — Run:AI training job launcher.
#
# This is NOT a deliverable script. It is a helper for longer compute runs
# where you want the job to refresh source code, execute a command, and exit.
# Training jobs can be preempted, so checkpoints and caches must live outside
# the source checkout under /scratch.

set -euo pipefail

# ============== EDIT THESE LINES ==============
RUN_NAME="${RUN_NAME:-debug_fkl_overfit}"  # Used in W&B and the default Hydra run_name override.
GASPAR="${GASPAR:-gaspar}"               # YOUR GASPAR EPFL username.
GROUP="${GROUP:-g67}"                     # YOUR TEAM, e.g. g07.
WANDB_MODE="${WANDB_MODE:-online}"       # online, offline, or disabled.

# Public source checkout. Each job fetches this branch and resets the checkout
# before running RUN_COMMAND, so new remote code is picked up automatically.
REPO_URL="${REPO_URL:-https://github.com/ouyangyuchen/knowledge-distill-for-spec-decoding.git}"
REPO_BRANCH="${REPO_BRANCH:-train}"
REPO_DIR="${REPO_DIR:-/scratch/cs552-repos/cs552-kdsd-${GASPAR}}"

# Persistent artifact locations outside the managed source checkout.
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/scratch/cs552-checkpoints}"
DATA_DIR="${DATA_DIR:-/scratch/cs552-data}"
HYDRA_OUTPUTS_DIR="${HYDRA_OUTPUTS_DIR:-/scratch/cs552-hydra-outputs}"
WANDB_DIR="${WANDB_DIR:-/scratch/wandb}"

# Edit this for your project. The command runs from REPO_DIR after checkout.
RUN_COMMAND="${RUN_COMMAND:-python scripts/train.py \
  loss=fkl data=ultrachat_10k \
  loss.alpha=1.0 \
  train.overfit_samples=16 \
  train.max_steps=300 \
  train.learning_rate=1e-4 \
  train.lr_scheduler_type=constant \
  train.warmup_ratio=0 \
  train.per_device_train_batch_size=1 \
  train.gradient_accumulation_steps=1 \
  run_name=debug_fkl_overfit16
}"
# ==============================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit submit_train.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit submit_train.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

if [[ "${RUN_COMMAND}" == *"<your-command>"* || -z "${RUN_COMMAND//[[:space:]]/}" ]]; then
    echo "ERROR: edit RUN_COMMAND before submitting a training job." >&2
    exit 1
fi

if [[ -z "${REPO_URL}" || -z "${REPO_BRANCH}" || -z "${REPO_DIR}" ]]; then
    echo "ERROR: REPO_URL, REPO_BRANCH, and REPO_DIR must all be set." >&2
    exit 1
fi

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: set WANDB_API_KEY before submitting, or run with WANDB_MODE=offline." >&2
    exit 1
fi

# Keep multiline shell payloads safe when they pass through the Run:AI CLI.
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

for required_name in REPO_URL REPO_BRANCH REPO_DIR WANDB_DIR CHECKPOINTS_DIR DATA_DIR HYDRA_OUTPUTS_DIR RUN_COMMAND_B64; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "ERROR: ${required_name} is empty inside the pod." >&2
    exit 1
  fi
done

RUN_COMMAND="$(printf '%s' "${RUN_COMMAND_B64}" | base64 -d)"

bootstrap_dirs=(
  /scratch/hf_cache
  /scratch/cs552-results
  /scratch/runs
  "${WANDB_DIR}"
  "${CHECKPOINTS_DIR}"
  "${DATA_DIR}"
  "${HYDRA_OUTPUTS_DIR}"
  "$(dirname "${REPO_DIR}")"
)

echo ">>> Ensuring persistent directories:"
printf '    %s\n' "${bootstrap_dirs[@]}"
mkdir -p "${bootstrap_dirs[@]}"

lock_file="${REPO_DIR}.lock"
exec 9>"${lock_file}"
if command -v flock >/dev/null 2>&1; then
  echo ">>> Waiting for managed checkout lock: ${lock_file}"
  flock 9
else
  echo ">>> WARNING: flock is unavailable; concurrent jobs using ${REPO_DIR} may race" >&2
fi

if command -v python3 >/dev/null 2>&1; then
  ln -sf "$(command -v python3)" /usr/local/bin/python 2>/dev/null || true
fi

configure_git_safe_directory() {
  # RCP /scratch is a shared PVC, so Git may reject repos there as "dubious
  # ownership". Configure only this managed checkout path as safe, and also set
  # env-based config so later Git commands work even if --global cannot write.
  export GIT_CONFIG_COUNT="${GIT_CONFIG_COUNT:-0}"
  local idx="${GIT_CONFIG_COUNT}"
  export "GIT_CONFIG_KEY_${idx}=safe.directory"
  export "GIT_CONFIG_VALUE_${idx}=${REPO_DIR}"
  export GIT_CONFIG_COUNT="$((idx + 1))"

  if git config --global --get-all safe.directory 2>/dev/null | grep -Fxq "${REPO_DIR}"; then
    return
  fi
  git config --global --add safe.directory "${REPO_DIR}" 2>/dev/null || true
}

configure_git_safe_directory

echo ">>> Checking git access to ${REPO_URL} branch ${REPO_BRANCH}"
if ! git ls-remote --exit-code --heads "${REPO_URL}" "${REPO_BRANCH}" >/dev/null; then
  echo "ERROR: Could not read branch '${REPO_BRANCH}' from ${REPO_URL}." >&2
  echo "For this public HTTPS repo, this usually means the branch name is wrong or the cluster cannot reach GitHub." >&2
  echo "Try overriding REPO_BRANCH=main or REPO_BRANCH=master if that is the branch you want." >&2
  exit 1
fi

if [[ -e "${REPO_DIR}" && ! -d "${REPO_DIR}" ]]; then
  echo "ERROR: REPO_DIR exists but is not a directory: ${REPO_DIR}" >&2
  exit 1
fi

if [[ -d "${REPO_DIR}" && ! -d "${REPO_DIR}/.git" ]]; then
  if find "${REPO_DIR}" -mindepth 1 -maxdepth 1 | read -r _; then
    echo "ERROR: REPO_DIR exists, is not a git checkout, and is not empty: ${REPO_DIR}" >&2
    echo "Move it aside or choose a different REPO_DIR." >&2
    exit 1
  fi
fi

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo ">>> Cloning ${REPO_URL} into ${REPO_DIR}"
  git clone --no-checkout --origin origin "${REPO_URL}" "${REPO_DIR}"
fi

configure_git_safe_directory

if ! git -C "${REPO_DIR}" rev-parse --git-dir >/dev/null 2>&1; then
  echo "ERROR: ${REPO_DIR} contains .git but Git cannot open it." >&2
  echo "This can happen after an interrupted clone. Move it aside or choose a different REPO_DIR." >&2
  exit 1
fi

current_origin="$(git -C "${REPO_DIR}" remote get-url origin 2>/dev/null || true)"
if [[ -z "${current_origin}" ]]; then
  echo ">>> Adding origin remote ${REPO_URL}"
  git -C "${REPO_DIR}" remote add origin "${REPO_URL}"
elif [[ "${current_origin}" != "${REPO_URL}" ]]; then
  echo ">>> Updating origin remote from '${current_origin}' to '${REPO_URL}'"
  git -C "${REPO_DIR}" remote set-url origin "${REPO_URL}"
fi

cd "${REPO_DIR}"

for artifact in checkpoints data outputs wandb; do
  if [[ -e "${artifact}" && ! -L "${artifact}" ]]; then
    echo "ERROR: ${REPO_DIR}/${artifact} exists and is not a symlink." >&2
    echo "This launcher will not delete real artifact directories. Move its contents to /scratch first." >&2
    exit 1
  fi
done

echo ">>> Fetching origin/${REPO_BRANCH}"
git fetch --prune origin "+refs/heads/${REPO_BRANCH}:refs/remotes/origin/${REPO_BRANCH}"
git clean -ffd
git checkout -B "${REPO_BRANCH}" "origin/${REPO_BRANCH}"
git reset --hard "origin/${REPO_BRANCH}"
git clean -ffd

link_artifact_dir() {
  local name="$1"
  local target="$2"

  if [[ -L "${name}" ]]; then
    rm "${name}"
  elif [[ -e "${name}" ]]; then
    echo "ERROR: ${REPO_DIR}/${name} exists and is not a symlink." >&2
    echo "Refusing to replace it with ${target}." >&2
    exit 1
  fi

  mkdir -p "${target}"
  ln -s "${target}" "${name}"
  echo ">>> Linked ${REPO_DIR}/${name} -> ${target}"
}

link_artifact_dir checkpoints "${CHECKPOINTS_DIR}"
link_artifact_dir data "${DATA_DIR}"
link_artifact_dir outputs "${HYDRA_OUTPUTS_DIR}"
link_artifact_dir wandb "${WANDB_DIR}"

commit_sha="$(git rev-parse HEAD)"
echo ">>> Checked out ${REPO_BRANCH} at ${commit_sha}"
echo ">>> Running command from ${REPO_DIR}:"
printf '%s\n' "${RUN_COMMAND}"
eval "${RUN_COMMAND}"
BOOTSTRAP

BOOTSTRAP_B64="$(printf '%s' "${BOOTSTRAP_COMMAND}" | base64 | tr -d '\n')"

echo ">>> Submitting training job ${JOB_NAME}  (1 GPU)"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu "${GPUS}" \
  --large-shm \
  --node-pools "${NODE}" \
  --working-dir /scratch \
  --environment HF_HOME=/scratch/hf_cache \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment WANDB_DIR="${WANDB_DIR}" \
  --environment WANDB_MODE="${WANDB_MODE}" \
  --environment WANDB_PROJECT="${WANDB_PROJECT:-cs552-kdsd}" \
  --environment WANDB_NAME="${RUN_NAME}" \
  --environment WANDB_ENTITY="${WANDB_ENTITY:-}" \
  --environment WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --environment REPO_URL="${REPO_URL}" \
  --environment REPO_BRANCH="${REPO_BRANCH}" \
  --environment REPO_DIR="${REPO_DIR}" \
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

This is a Run:AI training job, not an interactive Jupyter job. It exits
when RUN_COMMAND finishes. The launcher refreshes ${REPO_BRANCH} from
${REPO_URL} into ${REPO_DIR}, then links repo-local checkpoints/data/outputs/wandb
to persistent /scratch directories before running.
EOF
