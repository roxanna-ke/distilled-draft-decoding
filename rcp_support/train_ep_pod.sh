#!/bin/bash
# Runs inside the Run:AI pod after the repository has been prepared.

set -euo pipefail

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

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/scratch/cs552-mnlp-kzy}"
REPO_DIR="${REPO_DIR:-${WORKSPACE_ROOT}/repos/distilled-draft-decoding-train}"
HF_HOME_DIR="${HF_HOME_DIR:-${WORKSPACE_ROOT}/hf_cache}"
RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_ROOT}/results}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${WORKSPACE_ROOT}/checkpoints}"
DATA_DIR="${DATA_DIR:-${WORKSPACE_ROOT}/data}"
HYDRA_OUTPUTS_DIR="${HYDRA_OUTPUTS_DIR:-${WORKSPACE_ROOT}/hydra}"
WANDB_DIR="${WANDB_DIR:-${WORKSPACE_ROOT}/wandb}"
LINK_ARTIFACT_DIRS="${LINK_ARTIFACT_DIRS:-false}"

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
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-cs552-kdsd}"

cd "${REPO_DIR}"

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

for loss_name in ${LOSS_LIST}; do
  run_name="${loss_name}_${RUN_NAME_SUFFIX}"
  echo ">>> Starting training: loss=${loss_name} run_name=${run_name}"
  python scripts/train.py \
    loss="${loss_name}" data="${DATASET}" seed="${SEED}" run_name="${run_name}" \
    output_dir="${CHECKPOINTS_DIR}/${run_name}" \
    results_dir="${RESULTS_DIR}/${run_name}" \
    data_root="${DATA_DIR}" \
    hf_cache="${HF_HOME_DIR}" \
    hydra.run.dir="${HYDRA_OUTPUTS_DIR}/${run_name}" \
    model.target="${TARGET_MODEL}" model.draft_default="${DRAFT_MODEL}" \
    model.device=cuda model.dtype="${MODEL_DTYPE}" model.attn_impl="${ATTN_IMPL}" \
    model.trust_remote_code=false \
    data.response_source=original data.n_samples="${N_SAMPLES}" \
    data.val_samples="${VAL_SAMPLES}" data.eval_samples="${EVAL_SAMPLES}" \
    data.max_seq_len="${MAX_SEQ_LEN}" \
    data.hf_dataset.name="${HF_DATASET_NAME}" data.hf_dataset.split="${HF_DATASET_SPLIT}" \
    loss.alpha="${KD_ALPHA}" loss.temperature="${KD_TEMPERATURE}" \
    train.max_steps="${MAX_STEPS}" train.num_train_epochs=1 \
    train.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    train.per_device_eval_batch_size="${PER_DEVICE_EVAL_BATCH_SIZE}" \
    train.gradient_accumulation_steps="${GRAD_ACCUM}" \
    train.learning_rate="${LEARNING_RATE}" train.weight_decay="${WEIGHT_DECAY}" \
    train.warmup_ratio="${WARMUP_RATIO}" train.lr_scheduler_type="${LR_SCHEDULER_TYPE}" \
    train.logging_steps="${LOGGING_STEPS}" train.save_steps="${SAVE_STEPS}" \
    train.eval_steps="${EVAL_STEPS}" train.save_total_limit="${SAVE_TOTAL_LIMIT}" \
    ++train.load_best_model_at_end=true ++train.metric_for_best_model=eval_loss \
    ++train.greater_is_better=false ++train.save_best_model=true \
    train.bf16=true train.fp16=false train.gradient_checkpointing=true \
    train.dataloader_drop_last=true train.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
    train.remove_unused_columns=false train.report_to_wandb=true \
    train.resume_from_checkpoint="${RESUME_FROM_CHECKPOINT}" \
    train.overfit_samples="${OVERFIT_SAMPLES}" train.compile_target=false \
    eval.n_warmup=1 eval.n_repeats=3 eval.run_vanilla_baseline=true eval.write_generations=true \
    runtime.mode=sampling runtime.temperature=1.0 runtime.top_p=0.9 \
    runtime.gamma=4 runtime.max_new_tokens=256 \
    wandb.enabled=true wandb.project="${WANDB_PROJECT}" \
    wandb.dir="${WANDB_DIR}" wandb.mode="${WANDB_MODE}"
  echo ">>> Finished training: loss=${loss_name} run_name=${run_name}"
done
