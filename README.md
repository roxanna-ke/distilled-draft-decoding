# CS-552 Open Project: Knowledge Distillation for Speculative Decoding

This repository contains the current project code for training and evaluating
draft models for speculative decoding. The main target setup is a Qwen2.5
teacher/draft pair, with additional Qwen3 configs and vLLM-based evaluation
helpers for newer experiments.

The codebase is built around:

- canonical text-level dataset preparation from Hugging Face datasets
- optional target-generated response caching for response-source ablations
- KD training with `ce`, `fkl`, `rkl`, and `jsd`
- speculative decoding evaluation with per-step acceptance and timing metrics
- Hydra-based experiment configuration
- RunAI / RCP cluster launch scripts for interactive and batch workflows

## Repository Layout

```text
.
├── configs/                 Hydra config groups
│   ├── benchmark/
│   ├── data/
│   ├── eval/
│   ├── loss/
│   ├── model/
│   ├── runtime/
│   └── train/
├── src/kdsd/
│   ├── data/                dataset prep, tokenization, target generation
│   ├── eval/                HF runner, vLLM runner, metrics, benchmarks
│   ├── losses/              CE / FKL / RKL / JSD
│   ├── models/              target/draft loading
│   ├── sd/                  speculative decoding loop and instrumentation
│   ├── train/               trainer and callbacks
│   └── utils/
├── scripts/                 Python and shell experiment entrypoints
├── tests/                   unit tests
├── notebooks/               notebooks and interactive RunAI launcher
├── rcp_support/             cluster helpers
└── README.md
```



## Current Status

What is implemented and used:

- `scripts/prepare_data.py`
- `scripts/generate_target_responses.py`
- `scripts/train.py`
- `scripts/train_size.py`
- `scripts/evaluate_sd.py`
- `src/kdsd/**` package code for data, losses, training, HF eval, and vLLM eval
- unit tests under `tests/`

What exists but is currently just a placeholder:

- `scripts/runtime_sweep.py`
- `scripts/aggregate_results.py`

Do not rely on those two files until they are implemented.

## Default Models

The default Hydra config is `configs/config.yaml`, which currently composes:

- `model=qwen25`
- `data=ultrachat_50k`
- `loss=fkl`
- `train=default`
- `eval=default`
- `runtime=default`
- `benchmark=default`

The default Qwen2.5 model config in `configs/model/qwen25.yaml` uses:

- target: `Qwen/Qwen2.5-3B-Instruct`
- default draft: `Qwen/Qwen2.5-0.5B-Instruct`
- dtype: `bfloat16`
- device: `cuda`

There is also a `configs/model/qwen3.yaml` config for Qwen3 experiments:

- target: `Qwen/Qwen3-14B`
- default draft: `Qwen/Qwen3-0.6B`

## Environment

Project metadata lives in `pyproject.toml`.

- Python: `>=3.11,<3.13`
- package name: `kdsd`
- dependency manager: `uv`

Local development is for editing and unit tests only:

```bash
uv sync
uv run pytest -q
```

You will also need Hugging Face access for Qwen checkpoints:

```bash
huggingface-cli login
```

The repository assumes heavy runs happen on the EPFL RCP / RunAI cluster, not
on a laptop.

## RunAI / Cluster Scripts

Interactive GPU Jupyter pod:

- `notebooks/submit.sh`
- `rcp_support/submit.sh`

Interactive CPU-only Jupyter pod:

- `rcp_support/submit_cpu.sh`

Non-interactive batch job launcher:

- `rcp_support/submit_train.sh`

Cluster reference:

- `rcp_support/README.md`

Note on `scripts/env.sh`: it does not create a virtual environment. It selects
an existing one, preferring `/scratch/venvs/kdsd-vllm` when present, and falls
back to the current `python` otherwise.

## Data Pipeline

The canonical on-disk format is JSONL with one record per row:

```json
{"id":"...", "prompt_text":"...", "response_text":"...", "source":"..."}
```

By default, heavy data artifacts are written under:

- `/scratch/cs552-data/processed/...`
- `/scratch/cs552-data/target_generated/...`
- `/scratch/cs552-data/tokenized/...`

### 1. Prepare processed splits

Example:

```bash
python scripts/prepare_data.py data=ultrachat_50k
```

Available data configs under `configs/data/` currently include:

- `ultrachat_10k`
- `ultrachat_25k`
- `ultrachat_50k`
- `ultrachat_50k_target_gen`
- `alpaca_50k`
- `eval_holdout`

For `ultrachat_50k`, this writes:

- `/scratch/cs552-data/processed/ultrachat_50k/train.jsonl`
- `/scratch/cs552-data/processed/ultrachat_50k/val.jsonl`
- `/scratch/cs552-data/processed/ultrachat_50k/eval.jsonl`

### 2. Generate target responses

Example:

```bash
python scripts/generate_target_responses.py data=ultrachat_50k_target_gen
```

Current configs default target-response generation to the `vllm` backend. The
script also supports `hf` via `data.target_generation.backend=hf`.

For target-generated data, the source processed split must exist first. If you
run `scripts/train.py` on a target-generated config and the source split is
missing, training will auto-trigger data preparation and target generation.

## Training

Main entrypoint:

```bash
python scripts/train.py loss=fkl data=ultrachat_50k run_name=kd_fkl_50k
```

Example on target-generated responses:

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

Current training behavior worth knowing:

- if `cfg.data.train_path` is missing, `scripts/train.py` auto-runs data prep
- if `data.response_source=target_generated`, it can auto-run target generation
- CE-only training skips loading the frozen target model
- W&B reporting is enabled by default through config unless overridden

To disable W&B explicitly:

```bash
python scripts/train.py ... train.report_to_wandb=false wandb.enabled=false
```

There is also `scripts/train_size.py`, which is used by some ablation shell
scripts and has separate trainer logic/tests.

## Evaluation

Main evaluation entrypoint:

```bash
python scripts/evaluate_sd.py draft=pretrained run_name=pretrained_sd_eval
```

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

Current eval backends:

- `eval.backend=manual`: HF models + custom instrumented speculative decoding loop
- `eval.backend=vllm`: vLLM-based eval through `src/kdsd/eval/vllm_runner.py`

Default eval config in `configs/eval/default.yaml`:

- `backend=manual`
- `n_warmup=1`
- `n_repeats=3`
- `run_vanilla_baseline=true`

Default benchmark config in `configs/benchmark/default.yaml` is:

- `benchmarks: []`

That means quality scoring is skipped unless you opt into benchmarks.

Evaluation writes:

- `/scratch/cs552-results/<run_name>/eval_summary.json`
- `/scratch/cs552-results/<run_name>/generations.jsonl`
- `/scratch/cs552-results/<run_name>/timing.json`
- `/scratch/cs552-results/<run_name>/config.yaml`

The HF speculative decoding implementation lives in
`src/kdsd/sd/instrument.py`. The summary includes metrics such as:

- `acceptance_rate`
- `avg_accepted_tokens`
- `sd_time_s`
- `vanilla_time_s`
- `speedup`
- `tokens_per_second`

W&B eval logging is also supported when `wandb.enabled=true`.

## Useful Shell Helpers

There are several shell entrypoints for specific experiments:

- `scripts/run_loss_size_ablation.sh`
- `scripts/run_loss_size_ablation_10k.sh`
- `scripts/run_qwen3_loss_sweep.sh`
- `scripts/run_target_response_cache.sh`
- `scripts/submit_qwen3_loss_sweep.sh`
- `scripts/submit_qwen3_8b_loss_sweep.sh`
- `scripts/submit_target_response_cache.sh`
- `rcp_support/run_vllm_eval_sweep.sh`
- `rcp_support/submit_vllm_eval_sweep.sh`

These are workflow-specific wrappers around the Python entrypoints and current
cluster layout. Read them before reuse; several assume fixed `/scratch/...`
paths or an existing vLLM environment.

