#!/bin/bash
# CS-552 — submit an interactive RCP job with Jupyter Lab on CPU only.
#
# This is a CPU-only variant of rcp_support/submit.sh. It keeps the same
# scratch/shared PVC mounts and Jupyter setup, but does not request a GPU or
# pin the job to the A100 node pool.
#
# Usage:
#   ./submit_cpu.sh              # default: CPU interactive Jupyter job
#   ./submit_cpu.sh exp1         # same job, custom name suffix

set -euo pipefail

# ============== EDIT THESE LINES ==============
GASPAR="ke"              # <-- For local runs: your EPFL username.
GROUP="g67"              # <-- REQUIRED: your team, e.g. g07.
# ==============================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: set GASPAR to the EPFL username of the person launching this job." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: set GROUP to your team number (e.g. g07)." >&2
    echo "GROUP must be correct because it selects your team's scratch PVC." >&2
    exit 1
fi

SUFFIX="${1:-cpu-lab}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

echo ">>> Submitting ${JOB_NAME} (CPU only, no GPU requested)"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --large-shm \
  --interactive \
  --working-dir /scratch \
  --environment HF_HOME=/scratch/hf_cache \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment WANDB_DIR=/scratch/wandb \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    mkdir -p /scratch/hf_cache /scratch/wandb && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd /scratch && \
    jupyter lab \
      --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
      --ServerApp.root_dir=/scratch \
      --ServerApp.token=\${JUPYTER_TOKEN:-cs552}"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
When Running:      runai port-forward ${JOB_NAME} --port 8888:8888 -p ${PROJECT}
Then open:         http://localhost:8888  (token: cs552)
Shell in pod:      runai bash ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

Notes:
  - This variant does not request a GPU and does not pin to the A100 pool.
  - Good fit for editing, notebooks, config work, and lightweight unit tests.
  - Not suitable for this repo's training / inference / SD evaluation pipeline.

Storage inside the pod:
  /scratch             group scratch PVC (${SCRATCH_PVC}) — your group's primary workspace, RW
  /shared-ro/datasets  course datasets (read-only)
  /shared-ro/models    course models   (read-only)
  /shared-rw           shared with ALL students — be careful what you write
EOF
