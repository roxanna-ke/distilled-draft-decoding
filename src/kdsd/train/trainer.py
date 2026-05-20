"""Training loop for KD speculative-decoding draft models.

The trainer consumes the text-level JSONL contract from ``prepare_data.py`` and
does tokenization/masking at load time:

    input_ids = chat_template(prompt) + response + eos
    labels    = -100 on prompt tokens, response/eos token ids on response tokens

Losses are computed on shifted logits/labels, so the model is trained as a
standard next-token LM while ignoring prompt-only positions.
"""

from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from kdsd.losses import kd_loss
from kdsd.utils.io import read_jsonl, write_json
from kdsd.utils.logging import get_logger

LOG = get_logger("kdsd.train")
_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TrainResult:
    output_dir: Path
    model_dir: Path
    final_loss: float
    final_ce: float
    final_kd: float
    steps: int


class PromptResponseDataset(Dataset):
    """JSONL prompt/response dataset with Qwen-style chat formatting."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int,
        add_eos: bool = True,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must be >= 2")
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.add_eos = bool(add_eos)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        prompt = _format_prompt(self.tokenizer, str(row["prompt_text"]))
        response = str(row["response_text"])
        if self.add_eos and self.tokenizer.eos_token:
            response = response + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        response_ids = self.tokenizer(
            response, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        if not response_ids:
            # The data prep script filters empty strings, but tokenizers can
            # still produce an empty sequence for unusual control-only text.
            response_ids = [self.tokenizer.eos_token_id]

        prompt_ids, response_ids = _truncate_prompt_response(
            prompt_ids,
            response_ids,
            max_length=self.max_length,
        )
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class DataCollatorForKD:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id must be set")
        self.pad_token_id = int(tokenizer.pad_token_id)

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(int(f["input_ids"].numel()) for f in features)
        input_ids = torch.full(
            (len(features), max_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)

        for i, f in enumerate(features):
            n = int(f["input_ids"].numel())
            input_ids[i, :n] = f["input_ids"]
            labels[i, :n] = f["labels"]
            attention_mask[i, :n] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_kd(
    *,
    target: PreTrainedModel,
    draft: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: DictConfig,
    device: str,
    target_id: str,
    draft_id: str,
) -> TrainResult:
    """Train ``draft`` against ``target`` according to the Hydra config."""
    train_cfg = cfg.train
    loss_cfg = cfg.loss
    out_dir = _resolve_path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _load_rows(train_cfg.train_path)
    val_rows = _load_rows(train_cfg.val_path, required=False)
    if not train_rows:
        raise ValueError(f"no training rows found at {train_cfg.train_path}")

    train_ds = PromptResponseDataset(
        train_rows,
        tokenizer,
        max_length=int(train_cfg.max_length),
        add_eos=bool(train_cfg.add_eos),
    )
    val_ds = PromptResponseDataset(
        val_rows,
        tokenizer,
        max_length=int(train_cfg.max_length),
        add_eos=bool(train_cfg.add_eos),
    ) if val_rows else None

    collator = DataCollatorForKD(tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.batch_size),
        shuffle=True,
        num_workers=int(train_cfg.num_workers),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.batch_size),
        shuffle=False,
        num_workers=int(train_cfg.num_workers),
        collate_fn=collator,
    ) if val_ds is not None else None

    target.eval()
    target.requires_grad_(False)
    draft.train()
    if bool(train_cfg.gradient_checkpointing):
        draft.gradient_checkpointing_enable()
    if hasattr(draft.config, "use_cache"):
        draft.config.use_cache = False
    if hasattr(target.config, "use_cache"):
        target.config.use_cache = False

    optimizer = torch.optim.AdamW(
        draft.parameters(),
        lr=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.weight_decay),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(train_cfg.warmup_steps),
        num_training_steps=int(train_cfg.steps),
    )

    LOG.info(
        "training %s from %s against target=%s loss=%s alpha=%.3f temp=%.3f rows=%d",
        draft_id,
        train_cfg.draft_init,
        target_id,
        loss_cfg.kind,
        float(loss_cfg.alpha),
        float(loss_cfg.temperature),
        len(train_rows),
    )
    wandb_run = _init_wandb(cfg, output_dir=out_dir)

    final_metrics = {"loss": math.nan, "ce": math.nan, "kd": math.nan}
    global_step = 0
    try:
        optimizer.zero_grad(set_to_none=True)
        grad_accum = max(1, int(train_cfg.gradient_accumulation_steps))
        micro_step = 0
        log_buffer: list[dict[str, float]] = []
        train_iter = _infinite_loader(train_loader)
        progress = tqdm(total=int(train_cfg.steps), desc="train", dynamic_ncols=True)

        while global_step < int(train_cfg.steps):
            batch = _to_device(next(train_iter), device)
            metrics = _training_micro_step(
                target=target,
                draft=draft,
                batch=batch,
                loss_kind=str(loss_cfg.kind),
                temperature=float(loss_cfg.temperature),
                alpha=float(loss_cfg.alpha),
            )
            (metrics["loss"] / grad_accum).backward()

            log_buffer.append({
                "loss": float(metrics["loss"].detach().cpu()),
                "ce": float(metrics["ce"].detach().cpu()),
                "kd": float(metrics["kd"].detach().cpu()),
            })
            micro_step += 1

            if micro_step % grad_accum != 0:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                draft.parameters(), float(train_cfg.max_grad_norm)
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.update(1)

            final_metrics = _mean_metrics(log_buffer)
            log_buffer.clear()
            lr = float(scheduler.get_last_lr()[0])
            if global_step % int(train_cfg.log_steps) == 0:
                LOG.info(
                    "step=%d loss=%.4f ce=%.4f kd=%.4f lr=%.3e grad_norm=%.3f",
                    global_step,
                    final_metrics["loss"],
                    final_metrics["ce"],
                    final_metrics["kd"],
                    lr,
                    float(grad_norm),
                )
                _wandb_log(wandb_run, {
                    "train/loss": final_metrics["loss"],
                    "train/ce": final_metrics["ce"],
                    "train/kd": final_metrics["kd"],
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm),
                    "train/micro_step": micro_step,
                }, step=global_step)
            if val_loader is not None and global_step % int(train_cfg.eval_steps) == 0:
                val_metrics = evaluate_loss(
                    target=target,
                    draft=draft,
                    loader=val_loader,
                    cfg=cfg,
                    device=device,
                    max_batches=int(train_cfg.max_eval_batches),
                )
                LOG.info(
                    "eval step=%d loss=%.4f ce=%.4f kd=%.4f",
                    global_step,
                    val_metrics["loss"],
                    val_metrics["ce"],
                    val_metrics["kd"],
                )
                _wandb_log(wandb_run, {
                    "eval/loss": val_metrics["loss"],
                    "eval/ce": val_metrics["ce"],
                    "eval/kd": val_metrics["kd"],
                }, step=global_step)
            if global_step % int(train_cfg.save_steps) == 0:
                _save_checkpoint(
                    out_dir / f"checkpoint-{global_step}",
                    draft=draft,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    meta=_meta(
                        cfg=cfg,
                        target_id=target_id,
                        draft_id=draft_id,
                        metrics=final_metrics,
                        steps=global_step,
                    ),
                )

        progress.close()
        if log_buffer:
            final_metrics = _mean_metrics(log_buffer)
    finally:
        if "progress" in locals():
            progress.close()

    model_dir = out_dir / "model"
    _save_checkpoint(
        out_dir,
        draft=draft,
        tokenizer=tokenizer,
        cfg=cfg,
        meta=_meta(
            cfg=cfg,
            target_id=target_id,
            draft_id=draft_id,
            metrics=final_metrics,
            steps=global_step,
        ),
    )
    _wandb_log(wandb_run, {
        "train/final_loss": final_metrics["loss"],
        "train/final_ce": final_metrics["ce"],
        "train/final_kd": final_metrics["kd"],
    }, step=global_step)
    _finish_wandb(wandb_run, cfg=cfg, model_dir=model_dir)
    if bool(train_cfg.save_optimizer):
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": global_step,
            },
            out_dir / "optimizer.pt",
        )

    return TrainResult(
        output_dir=out_dir,
        model_dir=model_dir,
        final_loss=final_metrics["loss"],
        final_ce=final_metrics["ce"],
        final_kd=final_metrics["kd"],
        steps=global_step,
    )


@torch.inference_mode()
def evaluate_loss(
    *,
    target: PreTrainedModel,
    draft: PreTrainedModel,
    loader: DataLoader,
    cfg: DictConfig,
    device: str,
    max_batches: int,
) -> dict[str, float]:
    was_training = draft.training
    draft.eval()
    rows: list[dict[str, float]] = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = _to_device(batch, device)
        metrics = _forward_loss(
            target=target,
            draft=draft,
            batch=batch,
            loss_kind=str(cfg.loss.kind),
            temperature=float(cfg.loss.temperature),
            alpha=float(cfg.loss.alpha),
        )
        rows.append({
            "loss": float(metrics["loss"].detach().cpu()),
            "ce": float(metrics["ce"].detach().cpu()),
            "kd": float(metrics["kd"].detach().cpu()),
        })
    if was_training:
        draft.train()
    return _mean_metrics(rows)


def _training_micro_step(
    *,
    target: PreTrainedModel,
    draft: PreTrainedModel,
    batch: dict[str, torch.Tensor],
    loss_kind: str,
    temperature: float,
    alpha: float,
) -> dict[str, torch.Tensor]:
    return _forward_loss(
        target=target,
        draft=draft,
        batch=batch,
        loss_kind=loss_kind,
        temperature=temperature,
        alpha=alpha,
    )


def _forward_loss(
    *,
    target: PreTrainedModel,
    draft: PreTrainedModel,
    batch: dict[str, torch.Tensor],
    loss_kind: str,
    temperature: float,
    alpha: float,
) -> dict[str, torch.Tensor]:
    student_logits = draft(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    ).logits

    teacher_logits = None
    if loss_kind != "ce":
        with torch.no_grad():
            teacher_logits = target(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            ).logits

    # Causal LM alignment: logits at position t predict label at position t+1.
    labels = batch["labels"][:, 1:].contiguous()
    student_logits = student_logits[:, :-1, :].contiguous()
    teacher_logits = teacher_logits[:, :-1, :].contiguous() if teacher_logits is not None else None

    return kd_loss(
        student_logits,
        teacher_logits,
        None,
        None,
        labels,
        kind=loss_kind,
        temperature=temperature,
        alpha=alpha,
    )


def _format_prompt(tokenizer: PreTrainedTokenizerBase, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _truncate_prompt_response(
    prompt_ids: list[int],
    response_ids: list[int],
    *,
    max_length: int,
) -> tuple[list[int], list[int]]:
    response_keep = min(len(response_ids), max_length - 1)
    response_ids = response_ids[:response_keep]
    prompt_budget = max_length - response_keep
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:]
    return prompt_ids, response_ids


def _load_rows(path: str | Path, *, required: bool = True) -> list[dict[str, Any]]:
    p = _resolve_path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(p)
        return []
    rows = read_jsonl(p)
    required_keys = {"prompt_text", "response_text"}
    bad = [i for i, row in enumerate(rows[:100]) if not required_keys <= row.keys()]
    if bad:
        raise ValueError(f"{p} has rows missing {sorted(required_keys)}; first bad rows={bad[:5]}")
    return rows


def _save_checkpoint(
    out_dir: Path,
    *,
    draft: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: DictConfig,
    meta: dict[str, Any],
) -> None:
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    draft.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    write_json(out_dir / "meta.json", meta)
    LOG.info("saved checkpoint to %s", out_dir)


def _init_wandb(cfg: DictConfig, *, output_dir: Path):
    wandb_cfg = cfg.train.get("wandb")
    if not wandb_cfg or not bool(wandb_cfg.get("enabled", False)):
        return None
    import wandb

    mode = str(wandb_cfg.get("mode", "online"))
    os.environ.setdefault("WANDB_MODE", mode)
    project = str(wandb_cfg.get("project") or "kdsd")
    entity = wandb_cfg.get("entity")
    entity = None if entity in (None, "", "null") else str(entity)
    tags = list(wandb_cfg.get("tags") or [])
    tags.extend([str(cfg.loss.kind), str(cfg.data.id)])

    run = wandb.init(
        project=project,
        entity=entity,
        name=str(cfg.run_name),
        id=str(cfg.run_name),
        resume="allow",
        tags=tags,
        dir=str(output_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("eval/*", step_metric="step")
    LOG.info("wandb enabled: project=%s entity=%s run=%s", project, entity, cfg.run_name)
    return run


def _wandb_log(run, metrics: dict[str, float], *, step: int) -> None:
    if run is None:
        return
    metrics = {"step": int(step), **metrics}
    run.log(metrics, step=int(step))


def _finish_wandb(run, *, cfg: DictConfig, model_dir: Path) -> None:
    if run is None:
        return
    if bool(cfg.train.wandb.get("log_model", False)):
        import wandb

        artifact = wandb.Artifact(f"{cfg.run_name}-model", type="model")
        artifact.add_dir(str(model_dir))
        run.log_artifact(artifact)
    run.finish()


def _meta(
    *,
    cfg: DictConfig,
    target_id: str,
    draft_id: str,
    metrics: dict[str, float],
    steps: int,
) -> dict[str, Any]:
    return {
        "git_sha": _git_sha(),
        "target": target_id,
        "draft_init": draft_id,
        "dataset_id": str(cfg.data.id),
        "loss": OmegaConf.to_container(cfg.loss, resolve=True),
        "train_loss_final": metrics["loss"],
        "train_ce_final": metrics["ce"],
        "train_kd_final": metrics["kd"],
        "steps": int(steps),
    }


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"loss": math.nan, "ce": math.nan, "kd": math.nan}
    return {
        key: float(sum(row[key] for row in rows) / len(rows))
        for key in ("loss", "ce", "kd")
    }


def _resolve_path(path: str | Path) -> Path:
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else _ROOT / p


def _infinite_loader(loader: DataLoader):
    while True:
        yield from loader
