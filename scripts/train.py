"""Train the draft model with online target-forward KD."""

from __future__ import annotations

import inspect
import os
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.get("hf_cache"):
        hf_home = os.path.expanduser(str(cfg.hf_cache))
        Path(hf_home).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_HUB_CACHE"] = str(Path(hf_home) / "hub")
        os.environ["HF_DATASETS_CACHE"] = str(Path(hf_home) / "datasets")
    _run(cfg)


def _run(cfg: DictConfig) -> None:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed

    from kdsd.data import KDCollator, KDDataset
    from kdsd.models.loader import _resolve_dtype
    from kdsd.train import KDTrainer
    from kdsd.utils.experiment import ensure_run_name, git_sha, resolve_path
    from kdsd.utils.io import write_json
    from kdsd.utils.logging import get_logger

    log = get_logger("kdsd.train")
    run_name = ensure_run_name(cfg)
    if bool(cfg.train.report_to_wandb):
        os.environ["WANDB_NAME"] = run_name
    out_dir = resolve_path(str(cfg.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    log.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))
    log.info("output_dir=%s", out_dir)

    device = str(cfg.model.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA when model.device=cuda")

    train_path = resolve_path(str(cfg.data.train_path))
    val_path = resolve_path(str(cfg.data.val_path))
    _ensure_training_data(cfg, train_path=train_path, log=log)

    dtype = _resolve_dtype(str(cfg.model.dtype))

    tokenizer = AutoTokenizer.from_pretrained(
        str(cfg.model.target),
        trust_remote_code=bool(cfg.model.trust_remote_code),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    draft_init = str(cfg.train.draft_init)
    log.info("Loading draft=%s", draft_init)
    draft = AutoModelForCausalLM.from_pretrained(
        draft_init,
        dtype=dtype,
        attn_implementation=str(cfg.model.attn_impl),
        trust_remote_code=bool(cfg.model.trust_remote_code),
    )
    if bool(cfg.train.gradient_checkpointing):
        draft.gradient_checkpointing_enable()
        if hasattr(draft, "config"):
            draft.config.use_cache = False

    target = None
    if str(cfg.loss.kind).lower() != "ce":
        log.info("Loading frozen target=%s", cfg.model.target)
        target = AutoModelForCausalLM.from_pretrained(
            str(cfg.model.target),
            dtype=dtype,
            attn_implementation=str(cfg.model.attn_impl),
            trust_remote_code=bool(cfg.model.trust_remote_code),
        ).to(device)
        target.eval().requires_grad_(False)
        if bool(cfg.train.compile_target):
            if torch.cuda.is_available() and hasattr(torch, "compile"):
                log.info("Compiling target with torch.compile(mode='reduce-overhead')")
                target = torch.compile(target, mode="reduce-overhead")
            else:
                log.warning("train.compile_target=true ignored because CUDA/torch.compile is unavailable")
    else:
        log.info("Skipping frozen target load for CE-only training")

    train_ds = KDDataset(
        train_path,
        tokenizer,
        max_seq_len=int(cfg.data.max_seq_len),
        cache_dir=str(cfg.data.tokenized_cache_dir),
    )
    overfit_samples = int(cfg.train.get("overfit_samples", 0) or 0)
    if overfit_samples > 0:
        from torch.utils.data import Subset

        n_overfit = min(overfit_samples, len(train_ds))
        train_ds = Subset(train_ds, list(range(n_overfit)))
        log.warning("Debug overfit mode: restricted train dataset to %d examples", n_overfit)
    eval_ds = None
    if val_path.exists():
        eval_ds = KDDataset(
            val_path,
            tokenizer,
            max_seq_len=int(cfg.data.max_seq_len),
            cache_dir=str(cfg.data.tokenized_cache_dir),
        )
        if len(eval_ds) == 0:
            eval_ds = None
    log.info("Loaded train=%d eval=%s", len(train_ds), len(eval_ds) if eval_ds is not None else "none")

    args = _training_args(cfg, out_dir, TrainingArguments, do_eval=eval_ds is not None)
    trainer = KDTrainer(
        model=draft,
        target_model=target,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=KDCollator(tokenizer),
        tokenizer=tokenizer,
        kd_cfg=OmegaConf.to_container(cfg.loss, resolve=True),
    )
    result = trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)
    trainer.save_model(out_dir / "model")
    tokenizer.save_pretrained(out_dir / "model")
    OmegaConf.save(cfg, out_dir / "config.yaml")

    metrics = result.metrics if result is not None else {}
    wandb_meta = _wandb_run_metadata(enabled=bool(cfg.train.report_to_wandb))
    write_json(
        out_dir / "meta.json",
        {
            "git_sha": git_sha(),
            "run_name": run_name,
            "train_loss_final": metrics.get("train_loss"),
            "steps": metrics.get("global_step", cfg.train.max_steps),
            "dataset_id": str(cfg.data.id),
            "draft_init": draft_init,
            "target": str(cfg.model.target),
            "loss": OmegaConf.to_container(cfg.loss, resolve=True),
            "data": {
                "train_path": str(train_path),
                "val_path": str(val_path),
                "max_seq_len": int(cfg.data.max_seq_len),
            },
            "wandb": wandb_meta,
        },
    )
    log.info("Wrote checkpoint to %s", out_dir)


def _ensure_training_data(cfg: DictConfig, *, train_path: Path, log) -> None:
    """Ensure the configured text-level training split exists.

    The tokenized dataset class already handles its own cache. This helper is
    only concerned with the canonical JSONL split selected by cfg.data.
    """
    if train_path.exists():
        log.info("Using cached training data at %s", train_path)
        return

    data_id = str(cfg.data.get("id", "unknown"))
    response_source = str(cfg.data.get("response_source", "original"))
    log.warning(
        "Training data %s is missing; preparing data_id=%s response_source=%s",
        train_path,
        data_id,
        response_source,
    )

    if response_source == "target_generated":
        _ensure_source_processed_data(cfg, log=log)
        _run_target_response_generation(cfg)
    else:
        _run_prepare_data(cfg)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Expected training data at {train_path} after auto-preparation, but it was not written"
        )
    log.info("Prepared training data at %s", train_path)


def _ensure_source_processed_data(cfg: DictConfig, *, log) -> None:
    """Prepare base prompt/response JSONL before target-response generation."""
    gen_cfg = cfg.data.get("target_generation")
    if gen_cfg is None or not gen_cfg.get("source_processed_dir"):
        raise ValueError(
            "data.response_source=target_generated requires "
            "data.target_generation.source_processed_dir"
        )

    from kdsd.utils.experiment import resolve_path

    src_dir = resolve_path(str(gen_cfg.source_processed_dir))
    missing = [
        split
        for split in gen_cfg.get("splits", ["train", "val"])
        if not (src_dir / f"{split}.jsonl").exists()
    ]
    if not missing:
        return

    log.warning(
        "Base processed split(s) %s missing under %s; running scripts/prepare_data.py logic first",
        ", ".join(missing),
        src_dir,
    )
    _run_prepare_data(cfg)


def _run_prepare_data(cfg: DictConfig) -> None:
    from scripts.prepare_data import _run as prepare_data

    prepare_data(cfg)


def _run_target_response_generation(cfg: DictConfig) -> None:
    from scripts.generate_target_responses import _run as generate_target_responses

    generate_target_responses(cfg)


def _training_args(cfg: DictConfig, out_dir: Path, cls, *, do_eval: bool):
    train = cfg.train
    kwargs = {
        "output_dir": str(out_dir / "trainer_state"),
        "run_name": str(cfg.run_name),
        "per_device_train_batch_size": int(train.per_device_train_batch_size),
        "per_device_eval_batch_size": int(train.per_device_eval_batch_size),
        "gradient_accumulation_steps": int(train.gradient_accumulation_steps),
        "learning_rate": float(train.learning_rate),
        "weight_decay": float(train.weight_decay),
        "warmup_ratio": float(train.warmup_ratio),
        "lr_scheduler_type": str(train.lr_scheduler_type),
        "logging_steps": int(train.logging_steps),
        "save_steps": int(train.save_steps),
        "eval_steps": int(train.eval_steps),
        "save_total_limit": int(train.save_total_limit),
        "bf16": bool(train.bf16),
        "fp16": bool(train.fp16),
        "dataloader_drop_last": bool(train.dataloader_drop_last),
        "dataloader_num_workers": int(train.dataloader_num_workers),
        "remove_unused_columns": bool(train.remove_unused_columns),
        "save_safetensors": True,
        "report_to": ["wandb"] if bool(train.report_to_wandb) else [],
        "seed": int(cfg.seed),
    }
    if int(train.max_steps) > 0:
        kwargs["max_steps"] = int(train.max_steps)
    else:
        kwargs["num_train_epochs"] = float(train.num_train_epochs)

    params = inspect.signature(cls.__init__).parameters
    strategy = "steps" if do_eval else "no"
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = strategy
    else:
        kwargs["evaluation_strategy"] = strategy
    if "use_cpu" in params and str(cfg.model.device) == "cpu":
        kwargs["use_cpu"] = True
    elif "no_cuda" in params and str(cfg.model.device) == "cpu":
        kwargs["no_cuda"] = True
    return cls(**kwargs)


def _wandb_run_metadata(*, enabled: bool) -> dict[str, str | None]:
    if not enabled:
        return {}
    try:
        import wandb
    except Exception:
        return {}

    run = getattr(wandb, "run", None)
    if run is None:
        return {}
    return {
        "id": getattr(run, "id", None),
        "name": getattr(run, "name", None),
        "project": getattr(run, "project", None),
        "entity": getattr(run, "entity", None),
        "url": getattr(run, "url", None),
    }


if __name__ == "__main__":
    main()
