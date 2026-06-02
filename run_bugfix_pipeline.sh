#!/usr/bin/env bash
# Re-run KD train + SD eval on the bug-fixed code (commit 66cb088, grad-accum fix).
# Single GPU, sequential. Designed to be left running detached (nohup/setsid).
#
# Matrix: objective {fkl, rkl, jsd} x alpha {0.5 (KD+CE), 1.0 (pure divergence)}
#   => 6 trainings (~80 min each) + 1 vanilla/pretrained-draft baseline eval
#      + 6 draft evals (~5 min each).  Total ~8-9 h.
#
# Launch (survives logout):
#   cd /net/inltitan2/scratch2/tzhu/open-project-m2-shallowseek
#   nohup bash run_bugfix_pipeline.sh > run_bugfix_pipeline.master.log 2>&1 &
#   tail -f run_bugfix_pipeline.master.log
#
# Per-stage logs land in logs_bugfix/. Checkpoints/results use a *_fix suffix so
# they do NOT clobber the buggy d109468 runs.

set -uo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
REPO=/net/inltitan2/scratch2/tzhu/open-project-m2-shallowseek
PY=/home/tzhu/conda/envs/eeg_denoise_39/bin/python
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/net/inltitan2.epfl.ch/scratch2/tzhu/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_NO_TF=1          # silence eager TensorFlow import in this env
DATA_ROOT=/net/inltitan2.epfl.ch/scratch2/tzhu/open-project-m2-shallowseek/data
PROMPTS=data/processed/ultrachat_50k/eval.jsonl
EVAL_BACKEND="${EVAL_BACKEND:-manual}"

cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }
mkdir -p logs_bugfix

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
run() { echo "[$(ts)] >>> $1"; }

# ── Pre-flight: confirm fixed code + deps present ────────────────────────────
run "git HEAD = $(git rev-parse --short HEAD)  ($(git log -1 --format=%s))"
"$PY" - <<'EOF' || { echo "MISSING PYTHON DEPS — install hydra-core omegaconf jsonlines into the env first"; exit 1; }
import importlib.util as u, sys
miss=[m for m in ("torch","transformers","hydra","omegaconf","jsonlines","accelerate") if not u.find_spec(m)]
sys.exit(1 if miss else 0)
EOF
echo "[$(ts)] env OK"

# ── Matrix definition: "objective alpha alphatag" ────────────────────────────
# alphatag is used only for clean run_name/dir suffixes (0.5 -> a05, 1.0 -> a1).
JOBS=(
  # "fkl 0.5 a05"
  "fkl 1.0 a1"
  # "rkl 0.5 a05"
  "rkl 1.0 a1"
  # "jsd 0.5 a05"
  "jsd 1.0 a1"
)

TRAIN_COMMON=(
  model.device=cuda
  train.max_steps=1400             # ~3 epochs at eff-batch 32 (44644 train / 32 ≈ 1395 steps/epoch)
  data.max_seq_len=512
  train.per_device_train_batch_size=4
  train.gradient_accumulation_steps=8
  train.learning_rate=1e-5
  train.eval_steps=300               # ~14 eval points to watch eval_loss for overfit (turn-up)
  train.report_to_wandb=false        # wandb not installed in this env (config default flipped to true upstream)
  data_root="$DATA_ROOT"
)
EVAL_COMMON=(
  model.device=cuda
  prompts.jsonl="$PROMPTS"
  prompts.limit=20
  runtime.gamma=4
  runtime.max_new_tokens=128
  eval.backend="$EVAL_BACKEND"
  eval.n_warmup=1
  eval.n_repeats=2
)

# ── Phase 1: train every (objective, alpha) cell ─────────────────────────────
for job in "${JOBS[@]}"; do
  read -r kind alpha atag <<< "$job"
  name="kd_${kind}_${atag}_fix"
  run "TRAIN ${kind} alpha=${alpha} -> checkpoints/${name}"
  "$PY" scripts/train.py "${TRAIN_COMMON[@]}" \
      loss="$kind" loss.alpha="$alpha" run_name="$name" \
      > "logs_bugfix/train_${kind}_${atag}.log" 2>&1
  echo "[$(ts)] train ${name} exit=$?"
  # Overfit watch: dump the eval_loss trajectory (every 300 steps). If it bottoms
  # out and turns back up while train loss keeps falling -> overfitting.
  echo "    eval_loss trajectory (${name}):"
  tr '\r' '\n' < "logs_bugfix/train_${kind}_${atag}.log" \
    | grep -oE "'eval_loss': [0-9.]+" | sed 's/^/      /' || true
done

# ── Phase 2: vanilla / pretrained-draft baseline (no KD) ─────────────────────
run "EVAL baseline draft=Qwen/Qwen2.5-0.5B-Instruct -> results/base_nokd_fix"
"$PY" scripts/evaluate_sd.py "${EVAL_COMMON[@]}" \
    draft=Qwen/Qwen2.5-0.5B-Instruct \
    run_name=base_nokd_fix results_dir=results/base_nokd_fix \
    > logs_bugfix/eval_base.log 2>&1
echo "[$(ts)] eval base exit=$?"

# ── Phase 3: eval each trained draft ─────────────────────────────────────────
for job in "${JOBS[@]}"; do
  read -r kind alpha atag <<< "$job"
  name="kd_${kind}_${atag}_fix"
  ckpt="checkpoints/${name}/model"
  if [ ! -f "${ckpt}/model.safetensors" ]; then
    echo "[$(ts)] SKIP eval ${name}: checkpoint missing (training failed?)"
    continue
  fi
  run "EVAL ${name} -> results/${name}_eval"
  "$PY" scripts/evaluate_sd.py "${EVAL_COMMON[@]}" \
      draft="$ckpt" \
      run_name="${name}_eval" results_dir="results/${name}_eval" \
      > "logs_bugfix/eval_${kind}_${atag}.log" 2>&1
  echo "[$(ts)] eval ${name} exit=$?"
done

# ── Summary ──────────────────────────────────────────────────────────────────
run "ALL DONE — summary"
SUMMARIES=(base_nokd_fix)
for job in "${JOBS[@]}"; do
  read -r kind alpha atag <<< "$job"
  SUMMARIES+=("kd_${kind}_${atag}_fix_eval")
done
for r in "${SUMMARIES[@]}"; do
  f="results/$r/eval_summary.json"
  if [ -f "$f" ]; then
    "$PY" - "$f" <<'EOF'
import json,sys
d=json.load(open(sys.argv[1]))
print(f"{d['model']:26s} acc={d['acceptance_rate']:.3f} "
      f"avg_acc={d['avg_accepted_tokens']:.2f} speedup={d['speedup']:.3f}x "
      f"tok/s={d['tokens_per_second']:.1f}")
EOF
  else
    echo "$r : MISSING (check logs_bugfix/)"
  fi
done
