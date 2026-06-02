#!/usr/bin/env bash
# KD train + SD eval for fkl at alpha=0.5 and alpha=1.0, both reported to W&B.
# Params mirror run_alpha.yaml (max_steps=8000, eff-batch 8 = bs2*ga4, lr=2e-5,
# save/eval every 2000, load_best_model_at_end). Only `loss.alpha` varies.
#
# Designed to run unattended for days (nohup). Each (train -> eval) pair shares
# ONE wandb run: train logs train/* live, eval resumes the same run via the
# wandb id stored in checkpoints/<name>/meta.json and logs eval/* metrics.
#
# Launch (survives logout):
#   cd /net/inltitan2/scratch2/tzhu/open-project-m2-shallowseek
#   nohup bash run_alpha.sh > run_alpha.master.log 2>&1 &
#   tail -f run_alpha.master.log
#
# Per-stage logs in logs_alpha/. Checkpoints: checkpoints/fkl_*_a{05,1}.
# Project: cs552-kdsd (https://wandb.ai/anthonyzhutianyi-epfl/cs552-kdsd).

set -uo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
REPO=/net/inltitan2/scratch2/tzhu/open-project-m2-shallowseek
PY=/home/tzhu/conda/envs/eeg_denoise_39/bin/python
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/net/inltitan2.epfl.ch/scratch2/tzhu/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_NO_TF=1
DATA_ROOT=/net/inltitan2.epfl.ch/scratch2/tzhu/open-project-m2-shallowseek/data
PROMPTS=data/processed/ultrachat_50k/eval.jsonl
EVAL_BACKEND="${EVAL_BACKEND:-manual}"

cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }
mkdir -p logs_alpha

# ── W&B auth + project (key lives in .env; never echoed) ─────────────────────
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export WANDB_PROJECT="${WANDB_PROJECT:-cs552-kdsd}"
export WANDB_DIR="${WANDB_DIR:-wandb}"
export WANDB_MODE="${WANDB_MODE:-online}"   # set to "offline" if the host loses internet
mkdir -p "$WANDB_DIR"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
run() { echo "[$(ts)] >>> $1"; }

# ── Pre-flight ───────────────────────────────────────────────────────────────
run "git HEAD = $(git rev-parse --short HEAD)  ($(git log -1 --format=%s))"
if [ -z "${WANDB_API_KEY:-}" ]; then echo "FATAL: WANDB_API_KEY not set (check .env)"; exit 1; fi
"$PY" - <<'EOF' || { echo "FATAL: missing python deps"; exit 1; }
import importlib.util as u, sys
miss=[m for m in ("torch","transformers","hydra","omegaconf","jsonlines","accelerate","wandb") if not u.find_spec(m)]
sys.exit(1 if miss else 0)
EOF
echo "[$(ts)] env OK  (WANDB_PROJECT=$WANDB_PROJECT  WANDB_MODE=$WANDB_MODE)"

# ── Training hyperparameters (mirror run_alpha.yaml; only alpha varies) ──────
TRAIN_COMMON=(
  loss=fkl
  model.device=cuda
  data.max_seq_len=512
  train.max_steps=8000
  train.per_device_train_batch_size=2
  train.gradient_accumulation_steps=4
  train.learning_rate=2e-5
  train.save_steps=2000
  train.eval_steps=2000
  train.save_total_limit=4
  train.load_best_model_at_end=true
  train.metric_for_best_model=eval_loss
  train.greater_is_better=false
  train.report_to_wandb=true
  data_root="$DATA_ROOT"
)
# Eval protocol (consistent across runs; edit here if you want more prompts).
EVAL_COMMON=(
  model.device=cuda
  prompts.jsonl="$PROMPTS"
  prompts.limit=64
  runtime.gamma=4
  runtime.max_new_tokens=256
  eval.backend="$EVAL_BACKEND"
  eval.n_warmup=1
  eval.n_repeats=2
  wandb.enabled=true
)

# ── alpha sweep: 0.5 (KD+CE) and 1.0 (pure divergence) ───────────────────────
ALPHAS=("0.5 a05" "1.0 a1")

for spec in "${ALPHAS[@]}"; do
  read -r alpha atag <<< "$spec"
  name="fkl_ultra50k_s8000_seq512_effbs8_${atag}"

  run "TRAIN fkl alpha=${alpha} -> checkpoints/${name}  (wandb run_name=${name})"
  "$PY" scripts/train.py "${TRAIN_COMMON[@]}" \
      loss.alpha="$alpha" run_name="$name" \
      > "logs_alpha/train_${atag}.log" 2>&1
  echo "[$(ts)] train ${name} exit=$?"
  echo "    eval_loss trajectory (${name}):"
  tr '\r' '\n' < "logs_alpha/train_${atag}.log" \
    | grep -oE "'eval_loss': [0-9.]+" | sed 's/^/      /' || true

  ckpt="checkpoints/${name}/model"
  if [ ! -f "${ckpt}/model.safetensors" ]; then
    echo "[$(ts)] SKIP eval ${name}: checkpoint missing (train failed -> see logs_alpha/train_${atag}.log)"
    continue
  fi
  run "EVAL ${name} (resumes wandb run, logs eval/*) -> results/${name}_eval"
  "$PY" scripts/evaluate_sd.py "${EVAL_COMMON[@]}" \
      draft="$ckpt" \
      run_name="${name}_eval" results_dir="results/${name}_eval" \
      > "logs_alpha/eval_${atag}.log" 2>&1
  echo "[$(ts)] eval ${name} exit=$?"
done

# ── Summary ──────────────────────────────────────────────────────────────────
run "ALL DONE — summary"
for spec in "${ALPHAS[@]}"; do
  read -r alpha atag <<< "$spec"
  f="results/fkl_ultra50k_s8000_seq512_effbs8_${atag}_eval/eval_summary.json"
  if [ -f "$f" ]; then
    "$PY" - "$f" "$alpha" <<'EOF'
import json,sys
d=json.load(open(sys.argv[1])); a=sys.argv[2]
print(f"  alpha={a:<3} {d['model']:40s} acc={d['acceptance_rate']:.3f} "
      f"avg_acc={d['avg_accepted_tokens']:.2f} speedup={d['speedup']:.3f}x "
      f"tok/s={d['tokens_per_second']:.1f}")
EOF
  else
    echo "  alpha=${alpha}: MISSING (check logs_alpha/)"
  fi
done
echo "[$(ts)] W&B: https://wandb.ai/anthonyzhutianyi-epfl/${WANDB_PROJECT}"
