"""SD evaluation entrypoint — see CLAUDE.md §"Eval contract".

Loads HF target (+ optional draft), runs the instrumented loop in
`src/kdsd/sd/instrument.py`, and writes
`/scratch/cs552-results/<run_name>/{eval_summary.json, generations.jsonl,
timing.json, config.yaml}` (path comes from `cfg.results_dir`; the default
points at the RunAI group scratch PVC per rcp_support/README.md).

Override anything from the CLI, e.g.:

    python scripts/evaluate_sd.py \\
        draft=Qwen/Qwen2.5-0.5B-Instruct \\
        prompts.jsonl=data/processed/eval.jsonl \\
        prompts.limit=20 \\
        runtime.gamma=4 runtime.max_new_tokens=128 \\
        run_name=spec_smoke

The HF cache directory comes from `cfg.hf_cache` (defaults to /scratch/hf_cache
on the RunAI pod). It is exported into HF_HOME before transformers is
imported, since huggingface_hub reads HF_HOME at import time.
"""

from __future__ import annotations

import json
from numbers import Real
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from kdsd.utils.experiment import ensure_run_name

    # huggingface_hub freezes its cache paths at import time, so HF_HOME must
    # be set before any transformers/huggingface_hub import.
    if cfg.get("hf_cache"):
        hf_home = os.path.expanduser(str(cfg.hf_cache))
        Path(hf_home).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_HUB_CACHE"] = str(Path(hf_home) / "hub")
        os.environ["HF_DATASETS_CACHE"] = str(Path(hf_home) / "datasets")

    ensure_run_name(cfg)
    out_dir = Path(cfg.results_dir)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir

    _run_eval(cfg, out_dir)


def _run_eval(cfg: DictConfig, out_dir: Path) -> None:
    import random
    import numpy as np
    import torch

    from kdsd.eval.runner import run_hf_eval
    from kdsd.models.loader import load_pair
    from kdsd.utils.io import read_jsonl, validate_eval_summary, write_json, write_jsonl
    from kdsd.utils.logging import get_logger

    LOG = get_logger("kdsd.evaluate_sd")
    checkpoint_meta_path, checkpoint_meta = _checkpoint_metadata_from_draft(cfg.get("draft"))
    LOG.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))
    LOG.info("HF_HOME=%s", os.environ.get("HF_HOME"))
    if checkpoint_meta_path is not None:
        LOG.info("Using checkpoint metadata from %s", checkpoint_meta_path)

    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pair = load_pair(
        target_id=str(cfg.model.target),
        draft_spec=(None if cfg.draft in (None, "", "null") else str(cfg.draft)),
        dtype=str(cfg.model.dtype),
        device=str(cfg.model.device),
        attn_impl=str(cfg.model.attn_impl),
        trust_remote_code=bool(cfg.model.trust_remote_code),
        draft_default=str(cfg.model.get("draft_default") or "") or None,
    )
    LOG.info(
        "Loaded target=%s draft=%s on %s (dtype=%s)",
        pair.target_id, pair.draft_id, pair.device, pair.dtype,
    )

    prompts = _load_prompts(cfg, read_jsonl, LOG)
    LOG.info("Loaded %d prompts", len(prompts))

    summary, rows = run_hf_eval(
        target=pair.target,
        draft=pair.draft,
        tokenizer=pair.tokenizer,
        prompts=prompts,
        runtime=OmegaConf.to_container(cfg.runtime, resolve=True),  # type: ignore[arg-type]
        eval_cfg=OmegaConf.to_container(cfg.eval, resolve=True),    # type: ignore[arg-type]
        device=pair.device,
        target_id=pair.target_id,
        draft_id=pair.draft_id,
        run_name=str(cfg.run_name),
        benchmarks=list(cfg.benchmark.get("benchmarks") or []),
    )

    validate_eval_summary(summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "eval_summary.json", summary)
    if cfg.eval.get("write_generations", True):
        write_jsonl(out_dir / "generations.jsonl", rows)
    _write_timing_json(out_dir, summary, write_json)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    if bool(cfg.get("wandb", {}).get("enabled", False)):
        _report_eval_to_wandb(
            cfg=cfg,
            summary=summary,
            out_dir=out_dir,
            checkpoint_meta=checkpoint_meta,
            checkpoint_meta_path=checkpoint_meta_path,
            log=LOG,
        )

    LOG.info("Wrote eval results to %s", out_dir)
    LOG.info(
        "acceptance_rate=%.3f avg_accepted_tokens=%.2f "
        "speedup=%.2fx tokens/s=%.1f",
        summary["acceptance_rate"], summary["avg_accepted_tokens"],
        summary["speedup"], summary["tokens_per_second"],
    )


def _write_timing_json(out_dir: Path, summary: dict, write_json) -> None:
    write_json(out_dir / "timing.json", {
        "sd_time_s": summary["sd_time_s"],
        "vanilla_time_s": summary["vanilla_time_s"],
        "tokens_per_second": summary["tokens_per_second"],
        "n_warmup": summary["n_warmup"],
        "n_repeats": summary["n_repeats"],
    })


def _checkpoint_metadata_from_draft(draft_spec) -> tuple[Path | None, dict]:
    if draft_spec in (None, "", "null"):
        return None, {}

    draft_path = Path(str(draft_spec)).expanduser()
    if not draft_path.is_absolute():
        draft_path = _ROOT / draft_path

    candidates = [
        draft_path.parent / "meta.json",
        draft_path / "meta.json",
    ]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if isinstance(meta, dict):
                return path, meta
    return None, {}


def _flatten_wandb_metrics(summary: dict) -> dict[str, Real]:
    metric_keys = (
        "acceptance_rate",
        "avg_accepted_tokens",
        "speedup",
        "tokens_per_second",
        "sd_time_s",
        "vanilla_time_s",
        "n_prompts",
        "n_warmup",
        "n_repeats",
    )
    metrics: dict[str, Real] = {}
    for key in metric_keys:
        value = summary.get(key)
        if _is_wandb_number(value):
            metrics[f"eval/{key}"] = value

    engines = summary.get("engines")
    if isinstance(engines, dict):
        for engine_name, engine_metrics in engines.items():
            if not isinstance(engine_metrics, dict):
                continue
            for key, value in engine_metrics.items():
                if _is_wandb_number(value):
                    metrics[f"eval/{engine_name}/{key}"] = value

    quality_score = summary.get("quality_score")
    if isinstance(quality_score, dict):
        for key, value in quality_score.items():
            if _is_wandb_number(value):
                metrics[f"eval/quality/{key}"] = value
    return metrics


def _is_wandb_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _report_eval_to_wandb(
    *,
    cfg: DictConfig,
    summary: dict,
    out_dir: Path,
    checkpoint_meta: dict,
    checkpoint_meta_path: Path | None,
    log,
) -> None:
    import wandb

    wandb_cfg = cfg.wandb
    train_wandb = checkpoint_meta.get("wandb") if isinstance(checkpoint_meta, dict) else None
    if not isinstance(train_wandb, dict):
        train_wandb = {}

    checkpoint_run_name = checkpoint_meta.get("run_name") if isinstance(checkpoint_meta, dict) else None
    run_name = str(checkpoint_run_name or cfg.run_name)
    run_id = train_wandb.get("id")
    if checkpoint_meta and not run_id:
        log.warning(
            "Checkpoint metadata has no wandb.id; logging eval under name=%s "
            "without guaranteed historical-run resume",
            run_name,
        )

    project = str(train_wandb.get("project") or wandb_cfg.project)
    entity = str(train_wandb.get("entity") or wandb_cfg.entity or "")
    init_kwargs = {
        "project": project,
        "name": run_name,
        "id": run_id,
        "resume": str(wandb_cfg.resume),
        "dir": str(_resolve_optional_path(wandb_cfg.dir)),
        "mode": str(wandb_cfg.mode),
        "config": _wandb_eval_config(
            cfg=cfg,
            summary=summary,
            out_dir=out_dir,
            checkpoint_run_name=checkpoint_run_name,
            checkpoint_meta_path=checkpoint_meta_path,
        ),
    }
    if entity:
        init_kwargs["entity"] = entity

    run = wandb.init(**init_kwargs)
    try:
        wandb.log(_flatten_wandb_metrics(summary))
    finally:
        if run is not None:
            wandb.finish()


def _resolve_optional_path(path_like) -> Path:
    path = Path(str(path_like)).expanduser()
    if path.is_absolute():
        return path
    return _ROOT / path


def _wandb_eval_config(
    *,
    cfg: DictConfig,
    summary: dict,
    out_dir: Path,
    checkpoint_run_name,
    checkpoint_meta_path: Path | None,
) -> dict:
    return {
        "eval_target": summary.get("target"),
        "eval_draft": summary.get("draft"),
        "eval_decoding": summary.get("decoding", {}),
        "eval_results_dir": str(out_dir),
        "eval_checkpoint_run_name": checkpoint_run_name,
        "eval_checkpoint_meta_path": str(checkpoint_meta_path) if checkpoint_meta_path else None,
        "eval_checkpoint_path": str(cfg.draft) if cfg.get("draft") is not None else None,
    }


def _load_prompts(cfg: DictConfig, read_jsonl, LOG):
    from kdsd.eval.runner import PromptRecord

    p = cfg.prompts
    records: list[PromptRecord] = []
    if p.get("jsonl"):
        path = Path(p.jsonl)
        if not path.is_absolute():
            path = _ROOT / path
        for i, row in enumerate(read_jsonl(path)):
            records.append(
                PromptRecord(
                    id=str(row.get("id", i)),
                    prompt_text=row["prompt_text"],
                    response_text=row.get("response_text"),
                    source=row.get("source"),
                )
            )
    elif p.get("hf_dataset"):
        from datasets import load_dataset
        spec = p.hf_dataset
        ds = load_dataset(spec["name"], split=spec.get("split", "train"))
        field = spec.get("prompt_field", "prompt")
        for i, row in enumerate(ds):
            records.append(
                PromptRecord(
                    id=str(row.get("id", i)),
                    prompt_text=row[field],
                    source=spec["name"],
                )
            )
    else:
        LOG.warning("No prompt source set; using a tiny built-in smoke list")
        for i, t in enumerate([
            "Explain the theory of relativity in simple terms.",
            "Write a Python function to compute the Fibonacci sequence.",
            "What are the main differences between TCP and UDP?",
            "Summarise the plot of Romeo and Juliet in one paragraph.",
        ]):
            records.append(PromptRecord(id=f"smoke-{i}", prompt_text=t, source="builtin"))

    limit = p.get("limit")
    if limit is not None and limit > 0:
        records = records[: int(limit)]
    return records


if __name__ == "__main__":
    main()
