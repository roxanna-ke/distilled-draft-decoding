[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/QDjEejvC)
# CS-552 Open Project: Knowledge Distillation for Speculative Decoding

This repository trains a small draft model for speculative decoding against a
larger Qwen2.5 target model.

The current codebase centers on:

- data preparation from Hugging Face datasets into canonical JSONL splits
- optional target-generated response data for response-source ablations
- KD training with `ce`, `fkl`, `rkl`, and `jsd` losses
- instrumented speculative-decoding evaluation with acceptance-rate and timing
  metrics
- Hydra-based experiment configuration and RunAI/RCP cluster support

## Current Layout

Actual implementation lives in `src/`, `scripts/`, and `configs/`. 

```text
.
├── configs/               Hydra config groups
│   ├── benchmark/
│   ├── data/
│   ├── eval/
│   ├── loss/
│   ├── model/
│   ├── runtime/
│   └── train/
├── src/kdsd/              Python package
│   ├── data/              dataset prep, tokenization, target generation
│   ├── eval/              eval runner and metrics
│   ├── losses/            CE / FKL / RKL / JSD KD losses
│   ├── models/            target/draft loading
│   ├── sd/                speculative decoding loop and instrumentation
│   ├── train/             HF Trainer subclass
│   └── utils/
├── scripts/
│   ├── prepare_data.py
│   ├── generate_target_responses.py
│   ├── train.py
│   ├── train_size.py
│   ├── evaluate_sd.py
│   ├── run_loss_size_ablation.sh
│   └── check_loss_convergence.py
├── tests/
├── notebooks/             team notebooks and RunAI launcher
├── rcp_support/           cluster submission helpers
├── code/                  course placeholder
└── report/                course placeholder
```

## Models and Defaults

The default model pair is defined in `configs/model/qwen25.yaml`:

- target: `Qwen/Qwen2.5-3B-Instruct`
- draft default: `Qwen/Qwen2.5-0.5B-Instruct`
- dtype: `bfloat16`
- device: `cuda`

The top-level Hydra config is `configs/config.yaml`. By default it composes:

- `data=ultrachat_50k`
- `loss=fkl`
- `train=default`
- `eval=default`
- `runtime=default`
- `benchmark=default`

Important config paths:

- `cfg.loss.temperature`: KD temperature used during training
- `cfg.runtime.temperature`: decoding temperature used during evaluation
- `cfg.draft`: draft model spec for eval
- `cfg.output_dir`: checkpoint output directory
- `cfg.results_dir`: evaluation artifact directory

Current defaults also enable W&B in config. If you do not want remote logging,
override both training and eval explicitly, for example:

```bash
train.report_to_wandb=false wandb.enabled=false
```

## Environment

Local development:

```bash
uv sync
uv run pytest -q
```

Project metadata is in `pyproject.toml`. The package name is `kdsd`, and the
project targets Python `>=3.11,<3.13`.

For cluster usage, see `rcp_support/README.md` and `notebooks/submit.sh`.

You will also need Hugging Face access for the Qwen checkpoints, typically via:

```bash
huggingface-cli login
```

## Data Pipeline

Processed data is written under `data_root`, which defaults to
`/scratch/cs552-data`.

### 1. Prepare canonical splits

```bash
python scripts/prepare_data.py data=ultrachat_50k
```

This writes:

- `/scratch/cs552-data/processed/ultrachat_50k/train.jsonl`
- `/scratch/cs552-data/processed/ultrachat_50k/val.jsonl`
- `/scratch/cs552-data/processed/ultrachat_50k/eval.jsonl`

Each row follows the text-level format:

```json
{"id":"...", "prompt_text":"...", "response_text":"...", "source":"..."}
```

### 2. Generate target responses for response-source ablations

```bash
python scripts/generate_target_responses.py data=ultrachat_50k_target_gen
```

This reads the processed base split and writes target-generated responses under:

- `/scratch/cs552-data/target_generated/ultrachat_50k/train.jsonl`
- `/scratch/cs552-data/target_generated/ultrachat_50k/val.jsonl`

The `ultrachat_50k_target_gen` config uses:

- `mode=greedy`
- `temperature=0.0`
- `top_p=1.0`

## Training

Main training entrypoint:

```bash
python scripts/train.py loss=fkl data=ultrachat_50k run_name=kd_fkl_50k
```

Example with JSD on target-generated responses:

```bash
python scripts/train.py \
  loss=jsd \
  data=ultrachat_50k_target_gen \
  loss.alpha=1.0 \
  loss.temperature=1.0 \
  train.max_steps=4000 \
  run_name=kd_jsd_target_gen
```

Training writes:

- `checkpoints/<run_name>/model/`
- `checkpoints/<run_name>/config.yaml`
- `checkpoints/<run_name>/meta.json`

Notable training behavior in the current code:

- if the configured training split is missing, `scripts/train.py` auto-runs
  `prepare_data.py`
- for `response_source=target_generated`, it can auto-run
  `generate_target_responses.py`
- CE-only training skips loading the frozen target model

## Evaluation

Main evaluation entrypoint:

```bash
python scripts/evaluate_sd.py draft=pretrained run_name=pretrained_sd_eval
```

Useful variants:

Vanilla target-only decoding:

```bash
python scripts/evaluate_sd.py draft=null run_name=vanilla_eval
```

Evaluate a trained checkpoint:

```bash
python scripts/evaluate_sd.py \
  draft=checkpoints/kd_jsd_target_gen/model \
  prompts.jsonl=/scratch/cs552-data/processed/ultrachat_50k/eval.jsonl \
  prompts.hf_dataset=null \
  prompts.limit=50 \
  runtime.mode=sampling \
  runtime.temperature=1.0 \
  runtime.top_p=0.9 \
  runtime.gamma=4 \
  runtime.max_new_tokens=256 \
  run_name=kd_jsd_eval
```

Evaluation writes:

- `/scratch/cs552-results/<run_name>/eval_summary.json`
- `/scratch/cs552-results/<run_name>/generations.jsonl`
- `/scratch/cs552-results/<run_name>/timing.json`
- `/scratch/cs552-results/<run_name>/config.yaml`

The instrumented speculative decoding loop lives in `src/kdsd/sd/instrument.py`.
The summary includes:

- `acceptance_rate`
- `avg_accepted_tokens`
- `sd_time_s`
- `vanilla_time_s`
- `speedup`
- `tokens_per_second`

By default, benchmark scoring is disabled via `configs/benchmark/default.yaml`
with `benchmarks: []`.

## Testing

The repository already includes unit tests for configs, losses, data helpers,
trainer logic, and eval/W&B plumbing.

Run all tests:

```bash
uv run pytest -q
```

Run a focused subset:

```bash
uv run pytest -q tests/test_losses.py tests/test_trainer.py
```


