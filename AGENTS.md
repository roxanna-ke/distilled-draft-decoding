# Project Reference

This file is the living design + module-API reference for this repository. New
contributors and assistants should read it before adding code. It is intentionally
prescriptive: the pipeline shape and inter-module contracts must stay stable so that
ablations across **KD objective**, **training data**, and **runtime decoding** can be
combined without bespoke glue.

The project distills a small draft model so that `Qwen/Qwen2.5-0.5B-Instruct` can
serve as a high-acceptance speculative-decoding (SD) draft for
`Qwen/Qwen2.5-3B-Instruct`. The two models share a tokenizer (verified) and are
FP16/BF16 friendly.

All SD timing and per-step metrics come from a single HF-based instrumented loop
(`src/kdsd/sd/instrument.py`). The loop implements rejection sampling and
`DynamicCache` management directly so per-step `accepted_lens` is exposed —
something HF's `model.generate(assistant_model=draft)` does not surface, and
something vLLM 0.11+ no longer supports for arbitrary KD-trained drafts (V1
removed draft-model SD; only ngram/EAGLE/Medusa/MTP remain).

---

## Stack & Environment

- **Python 3.11**, `torch>=2.4`, `transformers>=4.45` (Qwen2.5 support).
- **`uv`** is the local dependency manager for development and unit tests.
  `pyproject.toml` + committed `uv.lock` are the single source of truth — do
  not introduce `requirements.txt`. Experiment commands run inside the RunAI pod
  with the pod's Python environment directly (`python scripts/...`), not
  `uv run`.
- Core deps: `torch`, `transformers>=4.45`, `accelerate`, `datasets`, `peft`
  (optional LoRA path), `hydra-core`, `omegaconf`, `wandb` (optional, gated by env
  var), `jsonlines`, `rich`, `numpy`, `tqdm`, `safetensors`.
  Dev deps: `pytest`, `ruff`, `pre-commit`.
- HF auth: set `HF_TOKEN` in the environment (or `huggingface-cli login`). Models
  cache under `~/.cache/huggingface` by default; override with `HF_HOME`.
- GPU target: 1× 40GB A100 on the EPFL RCP RunAI cluster (course cap),
  `bf16` preferred. The course image (`registry.rcp.epfl.ch/course-cs-552/base-vllm:v1`)
  ships torch 2.8 + cu128, transformers 4.57, and JupyterLab 4.5.
  See `rcp_support/README.md` for cluster setup, submission, and storage layout.

Local unit-test bootstrap:

```bash
uv sync
uv run pytest -q
```

RunAI pod sanity check (inside `runai bash <job-name>` or a Jupyter terminal):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Local dev policy.** Mac/Windows hosts are for code editing and **unit tests only**
(`uv sync`, then `uv run pytest -q`). No model inference, training, data
preparation, or evaluation runs locally — all of those go through RunAI on the
EPFL RCP cluster. See `rcp_support/README.md` for cluster setup, and
`notebooks/run_eval_pipeline.ipynb` for the interactive eval driver.

---

## Training Pipeline Decisions

The maintained training path is **online target forward**, not offline target-logit
caching. This follows the dominant pattern in KD-for-SD work such as DistillSpec,
MiniLLM, DistillM, SKD, Online Speculative Decoding, and FastDraft alignment:
on-policy variants and sequence-level mixing assume the teacher can be queried
during training. For this project, online teacher logits also avoid top-k cache
artifacts in the FKL/RKL/JSD comparison.

The one response cache we do maintain is **target-generated text**. Target
`model.generate()` is autoregressive and reused across data-ablation runs, so
`scripts/generate_target_responses.py` writes reusable text JSONL under
`/scratch/cs552-data/target_generated/<data_id>/`.

Storage layout for the training pipeline:

- `/scratch/cs552-data/processed/<data_id>/` — canonical text JSONL splits.
- `/scratch/cs552-data/target_generated/<data_id>/` — target-generated response text.
- `/scratch/cs552-data/tokenized/<fingerprint>/` — disposable tokenizer caches.
- `/scratch/cs552-results/<run_name>/` — eval outputs.

The canonical data format stays text-level:

```json
{"id": "...", "prompt_text": "...", "response_text": "...", "source": "...", "metadata": {}}
```

Training tokenization is cached by tokenizer, chat template, max sequence length,
source-file hash, and masking policy. These caches are an optimization only; delete
them freely when preprocessing changes.

---

## Directory Structure

The internal package is named `kdsd` ("knowledge distillation for speculative
decoding") — descriptive and neutral.

```
cs552-mnlp-project/
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── README.md
├── AGENTS.md
├── configs/                        # Hydra configs (composable)
│   ├── config.yaml
│   ├── model/
│   │   └── qwen25.yaml
│   ├── data/
│   │   ├── ultrachat_10k.yaml
│   │   ├── ultrachat_25k.yaml
│   │   ├── ultrachat_50k.yaml
│   │   ├── ultrachat_50k_target_gen.yaml
│   │   ├── alpaca_50k.yaml
│   │   └── eval_holdout.yaml
│   ├── loss/
│   │   ├── ce.yaml
│   │   ├── fkl.yaml
│   │   ├── rkl.yaml
│   │   └── jsd.yaml
│   ├── train/
│   │   └── default.yaml
│   ├── eval/
│   │   ├── default.yaml
│   │   └── runtime_sweep.yaml
│   ├── runtime/
│   │   ├── default.yaml
│   │   └── sweep.yaml
│   └── benchmark/
│       ├── default.yaml
│       └── full.yaml
├── src/kdsd/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── kd_pair.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py
│   │   ├── process.py
│   │   ├── target_generate.py
│   │   ├── logit_cache.py
│   │   └── dataset.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── ce.py
│   │   ├── fkl.py
│   │   ├── rkl.py
│   │   ├── jsd.py
│   │   └── combined.py
│   ├── sd/
│   │   ├── __init__.py
│   │   ├── hf_assisted.py
│   │   ├── custom_loop.py
│   │   └── instrument.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── runner.py
│   │   ├── metrics.py
│   │   └── benchmarks/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── judge_gpt4.py
│   │       ├── mt_bench.py
│   │       └── registry.py
│   ├── train/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── callbacks.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── timing.py
│       └── logging.py
├── scripts/
│   ├── prepare_data.py
│   ├── generate_target_responses.py
│   ├── cache_target_logits.py
│   ├── train.py
│   ├── evaluate_sd.py            # single-phase HF eval entrypoint
│   ├── hf_sd_speedup.py          # standalone HF SD speedup probe
│   ├── runtime_sweep.py
│   └── aggregate_results.py
├── tests/
│   ├── __init__.py
│   ├── test_losses.py
│   ├── test_data.py
│   ├── test_eval_schema.py
│   └── test_config_compose.py
├── notebooks/                    # RunAI deliverable layout
│   ├── submit.sh                 # team-wide Jupyter launcher (copy of rcp_support/submit.sh)
│   ├── run_eval_pipeline.ipynb   # interactive driver for scripts/evaluate_sd.py
│   └── <first>_<last>_<sciper>.ipynb  # one per teammate (deliverable)
├── rcp_support/                  # upstream RunAI / RCP starter (do not modify)
│   ├── README.md                 # canonical cluster guide
│   ├── submit.sh                 # interactive Jupyter starter
│   ├── submit_train.sh           # non-interactive training-job starter
│   ├── Dockerfile                # optional custom-image template
│   └── build.sh                  # optional Harbor push helper
└── (gitignored) data/, checkpoints/, wandb/   # results land in /scratch/cs552-results/
```

---

## Module Contracts

These are the only interfaces other modules may depend on. Internal helpers in each
subpackage stay private.

### 1. Checkpoint contract

Every training run writes:

```
checkpoints/<run_name>/
  config.yaml   # the resolved Hydra config (OmegaConf.save)
  model/        # safetensors + tokenizer (HF .save_pretrained format)
  meta.json     # {git_sha, train_loss_final, steps, dataset_id, draft_init}
```

Consequence: `evaluate_sd.py draft=checkpoints/<run_name>/model` works for **every**
checkpoint, with no special-casing per training recipe.

### 2. Eval contract — `scripts/evaluate_sd.py`

CLI for SD evaluation. One process per invocation; the HF instrumented loop in
`src/kdsd/sd/instrument.py` produces all timing and per-step metrics. Always
writes:

```
/scratch/cs552-results/<run_name>/
  eval_summary.json   # the schema below — used by aggregate_results.py
  generations.jsonl   # per-prompt: prompt, generation, accepted_lens[], times{}
  timing.json         # raw per-prompt timings, n_warmup, n_repeats
  config.yaml         # resolved config snapshot
```

The path is configurable via `cfg.results_dir`; the default lands on the
RunAI group scratch PVC per `rcp_support/README.md` §"Storage layout".

`eval_summary.json` schema (frozen top-level keys, plus an `engines.hf` block).
`quality_score` is a **dict** so a single config can be scored against multiple
benchmarks in one run:

```json
{
  "model": "kd_jsd_50k_target_gen",
  "target": "Qwen/Qwen2.5-3B-Instruct",
  "draft":  "checkpoints/kd_jsd_50k_target_gen/model",
  "acceptance_rate": 0.52,
  "avg_accepted_tokens": 2.1,
  "vanilla_time_s": 123.4,
  "sd_time_s": 87.2,
  "speedup": 1.41,
  "tokens_per_second": 18.6,
  "quality_score": {
    "gpt4_judge_vs_target": 7.1,
    "mt_bench": 6.8,
    "exact_match_vs_target": 0.42
  },
  "decoding": {"mode": "greedy", "max_new_tokens": 256, "num_assistant_tokens": 4},
  "n_prompts": 200, "n_warmup": 2, "n_repeats": 3,
  "engines": {
    "hf": {"sd_time_s": 87.2, "vanilla_time_s": 123.4, "speedup": 1.41,
           "tokens_per_second": 18.6, "acceptance_rate": 0.52,
           "avg_accepted_tokens": 2.1, "n_outer_steps": 1240,
           "target_calls": 1240, "draft_calls": 4960,
           "draft_forward_s": 12.1, "target_forward_s": 71.4,
           "batched": false}
  }
}
```

`"quality_score": {}` is valid (skip benchmarks). Validation lives in
`tests/test_eval_schema.py` and `src/kdsd/utils/io.py`. `aggregate_results.py`
reads `quality_score.<key>`, so adding a benchmark never requires changing the
aggregator. `engines` is **optional** (not in `REQUIRED_SUMMARY_KEYS`) so older
summary files still validate; when present, each engine's sub-block must be a
dict.

`engine=hf` loads the HF target (and the optional draft), runs the
instrumented SD loop and a vanilla baseline (when `draft` is set and
`eval.run_vanilla_baseline=true`), and writes the full eval_summary +
generations + timing artefacts. The HF loop has unavoidable D→H syncs
(accept-mask transfer, EOS scan, rejection-resample `s>0` check) so its
`speedup` is conservative — but it is consistent across all KD ablations,
which is what the attribution table cares about.

Run an eval:

```bash
python scripts/evaluate_sd.py \
    run_name=spec_smoke draft=Qwen/Qwen2.5-0.5B-Instruct \
    prompts.jsonl=data/processed/eval.jsonl prompts.limit=20
```

### 3. Loss contract — `src/kdsd/losses/combined.py`

```python
def kd_loss(
    student_logits: Tensor,         # [B, T, V]
    teacher_logits: Tensor | None,  # None when using cached top-k path
    teacher_topk_ids: Tensor | None,
    teacher_topk_logp: Tensor | None,
    labels: Tensor,                 # -100 on prompt tokens
    *, kind: Literal["fkl","rkl","jsd","ce"],
    temperature: float = 1.0,
    alpha: float = 0.5,             # weight on KD term; (1-alpha) * CE
) -> dict[str, Tensor]:             # {"loss", "ce", "kd"}
```

Other modules import only `kd_loss`.

### 4. SD instrumentation contract — `src/kdsd/sd/instrument.py`

```python
@dataclass
class SDStats:
    accepted_lens: list[int]    # per generation step
    target_calls: int
    draft_calls: int
    draft_forward_s: float
    target_forward_s: float

def speculative_generate(
    target, draft, input_ids, *, gamma: int, max_new: int, ...
) -> tuple[Tensor, SDStats]
```

This is the source of truth for acceptance-rate / runtime-profile metrics. The
GPU-only forward times use CUDA events so the caller's wall-clock isn't
contaminated by per-forward sync overhead.

### 5. Data contract — processed split format

Every processed/target-generated split is a `.jsonl` with one record per row:

```json
{"id": "...", "prompt_text": "...", "response_text": "...", "source": "ultrachat|target"}
```

KD training re-tokenizes and masks at load time; this keeps text-level data
inspectable and decouples the on-disk format from the tokenizer in use.

### 6. Benchmark contract — `src/kdsd/eval/benchmarks/base.py`

```python
class Benchmark(ABC):
    name: str   # key in quality_score dict
    @abstractmethod
    def score(
        self, generations: list[dict], target_generations: list[dict] | None
    ) -> float: ...
```

`registry.py` maps name → class. The `benchmark/*.yaml` Hydra group lists which
benchmarks to run; missing API keys (e.g. for the GPT-4 judge) skip that benchmark
with a warning rather than failing the eval.

---

## Hydra Config Layout

`configs/config.yaml` is the top-level defaults list reused by every script:

```yaml
defaults:
  - model: qwen25
  - data: ultrachat_50k
  - loss: fkl
  - train: default
  - eval: default
  - runtime: default
  - benchmark: default
  - _self_

run_name: ${loss}_${data}_seed${seed}
seed: 42
output_dir: checkpoints/${run_name}
results_dir: /scratch/cs552-results/${run_name}
hf_cache: ${oc.env:HF_HOME,'~/.cache/huggingface'}
```

Every ablation is one CLI flip. All commands run **inside the RunAI pod**
with plain `python` — either from a Jupyter terminal, `runai bash <job>`, or
as the `TRAIN_COMMAND` of `rcp_support/submit_train.sh` for unattended runs.
From your laptop, launch the pod with `notebooks/submit.sh` (see
`rcp_support/`).

| Ablation              | Command                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| KD-objective sweep    | `python scripts/train.py -m loss=fkl,rkl,jsd`                            |
| Data-scale sweep      | `python scripts/train.py -m data=ultrachat_10k,ultrachat_25k,ultrachat_50k` |
| Response source       | `python scripts/train.py data=ultrachat_50k_target_gen`                  |
| Runtime sweep         | `python scripts/runtime_sweep.py runtime=sweep draft=checkpoints/<best>/model` |
| Multi-benchmark eval  | `python scripts/evaluate_sd.py benchmark=full draft=...`                 |
| Eval                  | `python scripts/evaluate_sd.py draft=...` (or `notebooks/run_eval_pipeline.ipynb`) |

---

## Pipeline — End-to-End

### Step 0: Launch / attach to the RunAI pod

From the laptop:

```bash
./notebooks/submit.sh exp1
runai bash <job-name>
```

Inside the pod, from the repo clone under `/scratch`:

```bash
cd /scratch/<repo>
huggingface-cli login   # store HF_TOKEN
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 1: Data

```bash
python scripts/prepare_data.py data=ultrachat_50k
python scripts/generate_target_responses.py data=ultrachat_50k    # optional arm
```

Writes:

- `/scratch/cs552-data/processed/ultrachat_50k/{train,val,eval}.jsonl`
- `/scratch/cs552-data/target_generated/ultrachat_50k/{train,val}.jsonl`
- `/scratch/cs552-data/tokenized/<fingerprint>/*.pt` when training first reads a split

### Step 2: Train

```bash
python scripts/train.py loss=jsd data=ultrachat_50k_target_gen \
  train.steps=4000 train.alpha=0.5 train.temperature=1.0 \
  run_name=kd_jsd_50k_targetgen_a0.5
```

Writes `checkpoints/kd_jsd_50k_targetgen_a0.5/{model,config.yaml,meta.json}`.

### Step 3: Evaluate

On RunAI, the **primary flow is the notebook**
`notebooks/run_eval_pipeline.ipynb`, which streams `evaluate_sd.py` output
inline and reads back the resulting `eval_summary.json`.

Headless / unattended use (e.g. `runai bash` or
`rcp_support/submit_train.sh`) keeps the equivalent CLI:

```bash
python scripts/evaluate_sd.py \
  draft=checkpoints/kd_jsd_50k_targetgen_a0.5/model \
  eval=default benchmark=default
```

End artefacts:
`/scratch/cs552-results/kd_jsd_50k_targetgen_a0.5/{eval_summary.json,generations.jsonl,timing.json}`.

### Step 4: Runtime sweep

```bash
python scripts/runtime_sweep.py \
  draft=checkpoints/<best>/model runtime=sweep
```

Iterates `gamma ∈ {1,2,4,6,8}` × `max_new ∈ {128,256}`, writing one
`/scratch/cs552-results/<best>__gamma{γ}_max{n}/eval_summary.json` per cell.

### Step 5: Aggregate

```bash
python scripts/aggregate_results.py /scratch/cs552-results/ \
  -o report/attribution_table.md
```

Joins every `eval_summary.json` into a staged attribution table — vanilla →
pretrained-draft SD → KD-adapted → data-adapted → runtime-tuned — with one column
per `quality_score.<benchmark>`.

---

## Cluster: RunAI / RCP

The project runs on the EPFL RCP RunAI cluster. `rcp_support/README.md` is
the canonical guide (one-time setup, port-forwarding, GPU etiquette,
storage). Two submission paths:

**Interactive Jupyter (primary, also the deliverable launcher).** From the
laptop:

```bash
./notebooks/submit.sh           # default suffix "lab"
./notebooks/submit.sh exp1      # custom job-name suffix
runai port-forward <job-name> --port 8888:8888
# open http://localhost:8888  (token: cs552)
```

This is a copy of `rcp_support/submit.sh` placed at the path the rcp_support
README mandates for the deliverable. It mounts `/scratch` (group PVC),
`/shared-ro`, `/shared-rw`, sets `HF_HOME=/scratch/hf_cache` and
`HF_HUB_ENABLE_HF_TRANSFER=1`, and starts JupyterLab in `/scratch`. Edit
`GASPAR` for your own runs and `GROUP` for your team — the
submitted file may keep `GASPAR="gaspar"` (TAs replace), but `GROUP` must be
correct because it selects your team's scratch PVC.

**Non-interactive training job.** For long runs that should execute a command
and exit, use `rcp_support/submit_train.sh` (do **not** submit it as a
deliverable). Set `TRAIN_COMMAND`, e.g. for an unattended eval:

```bash
TRAIN_COMMAND='cd /scratch/<repo> && python scripts/evaluate_sd.py \
  run_name=kd_jsd_50k draft=checkpoints/kd_jsd_50k/model'
./rcp_support/submit_train.sh
```

Training jobs are lower priority than interactive jobs and can be preempted —
write checkpoints to `/scratch` and resume from them.

**Storage contract.** Code lives in the git repo (cloned under `/scratch/`
inside the pod). HF cache and wandb logs live under `/scratch/hf_cache`
and `/scratch/wandb`. Deliverable notebooks must run from a clean clone of
the repo plus the course/group PVCs — no dependence on personal home or
ad-hoc files in `/scratch`. Anything in `/scratch` is wiped end of July 2026.

---

## File Build Order

When implementing the prototype, build in this order so downstream modules always
have a stable contract to call:

1. `pyproject.toml`, `.python-version`, `.gitignore`, `README.md`, `AGENTS.md`.
2. `src/kdsd/utils/{io,timing,logging}.py` — needed by everything.
3. `src/kdsd/models/{loader,kd_pair}.py` — load Qwen target/draft pair.
4. `src/kdsd/sd/instrument.py` — custom speculative-decoding loop with rejection
   sampling and KV-cache (`DynamicCache`) management. Implemented directly
   rather than going through HF's `model.generate(assistant_model=draft)` path
   because the eval_summary.json schema requires per-step `accepted_lens`,
   which HF's assisted decoding does not surface cleanly.
   `hf_assisted.py` / `custom_loop.py` remain as optional alternative entrypoints.
5. `src/kdsd/eval/{runner,metrics}.py` + `src/kdsd/eval/benchmarks/{base,registry}.py`
   - `scripts/evaluate_sd.py` — the contract everyone depends on; build first.
6. `src/kdsd/data/{download,process,target_generate,logit_cache,dataset}.py`
   - `scripts/prepare_data.py`, `scripts/generate_target_responses.py`,
   `scripts/cache_target_logits.py`.
7. `src/kdsd/losses/{ce,fkl,rkl,jsd,combined}.py` + `tests/test_losses.py`.
8. `src/kdsd/train/{trainer,callbacks}.py` + `scripts/train.py`.
9. `src/kdsd/eval/benchmarks/{judge_gpt4,mt_bench}.py`.
10. `scripts/runtime_sweep.py`, `scripts/aggregate_results.py`.
11. `configs/**` — fill once `src/` is stable, then add ablation YAMLs cheaply.
12. `notebooks/{submit.sh, run_eval_pipeline.ipynb}` last, after the Python
    entrypoints work. `rcp_support/` is provided upstream — do not modify;
    `notebooks/submit.sh` is a verbatim copy with `GROUP` filled in.

External libraries leaned on rather than reimplemented:

- HF `DynamicCache` for KV-cache management inside our custom SD loop.
  (HF's `model.generate(assistant_model=draft, num_assistant_tokens=γ)` is a
  fine fallback path, but we use the custom loop to expose `accepted_lens`.)
- HF `Trainer` with a `compute_loss` override for KD.
- `datasets.load_dataset` for UltraChat / Alpaca download + caching.
- `accelerate` for mixed-precision and (later) multi-GPU.

---

## Verification

**Local (Mac/Windows) — unit tests only.** No model inference is exercised locally.

```bash
uv sync
uv run pytest -q
```

Required passing tests:

- `tests/test_losses.py` — FKL ≠ RKL on asymmetric distributions; JSD symmetric;
  CE-only matches masked NLL.
- `tests/test_data.py` — prompt masking sets `labels = -100` on prompt tokens and
  leaves response tokens intact.
- `tests/test_eval_schema.py` — fixture `eval_summary.json` with the dict-form
  `quality_score` validates; missing required field is rejected.
- `tests/test_config_compose.py` — every advertised Hydra combination
  (every `loss` × every `data`) resolves without error.

**Cluster — end-to-end smoke (one short job, ~10 min on A100).**

From the laptop, launch the interactive pod:

```bash
./notebooks/submit.sh smoke
runai port-forward <job-name> --port 8888:8888
```

Inside Jupyter (token `cs552`), in a Jupyter terminal under
`/scratch/<repo>`:

```bash
python scripts/train.py run_name=smoke loss=fkl data=ultrachat_10k
```

Then open `notebooks/run_eval_pipeline.ipynb`, set
`RUN_NAME=smoke`, `DRAFT=checkpoints/smoke/model`, `Run All`. Delete the
job afterwards: `runai delete job <job-name>`.

Smoke acceptance:

- Training writes a loadable checkpoint and `meta.json`.
- Eval writes a schema-valid `eval_summary.json` for both `draft=null` (vanilla) and
  the trained draft, with finite, positive `speedup` and a populated
  `quality_score` dict.
- `aggregate_results.py` over a 2-row `/scratch/cs552-results/` produces a markdown table.

Once cluster smoke is green, the documented ablation commands above are ready.

---

## Conventions

- Result identifiers (`run_name`) flow from Hydra and are reused as both the
  `checkpoints/<run_name>/` (in-repo, gitignored) and
  `/scratch/cs552-results/<run_name>/` (group scratch PVC) directory names. Pick
  names that round-trip with `aggregate_results.py`.
- Never commit `data/`, `checkpoints/`, or `wandb/`. Eval artefacts live under
  `/scratch/cs552-results/` on the RunAI pod and are never staged into the repo.
- Schema changes to `eval_summary.json` are breaking — bump a `schema_version` field
  and update `tests/test_eval_schema.py` + `aggregate_results.py` in the same PR.
- Hydra overrides go on the CLI, not in code. If you find yourself hard-coding a
  hyperparameter inside a script, it belongs in a config group.
