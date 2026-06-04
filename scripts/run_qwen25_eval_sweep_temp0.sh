#!/bin/bash
# Temperature-0 variant of the Qwen2.5 eval sweep for ablations.

set -euo pipefail

export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_0p5b_ultrachat_50k_seed42_temp0}"
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qwen25_0p5b_temp0}"
export WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"
export HYDRA_ROOT="${HYDRA_ROOT:-/scratch/cs552-mnlp-kzy/hydra/qwen25-eval-sweep-temp0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_qwen25_eval_sweep_temp1.sh"
