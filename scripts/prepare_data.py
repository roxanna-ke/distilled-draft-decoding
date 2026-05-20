"""Prepare text-level train/val/eval JSONL splits for KD draft training.

The on-disk contract is intentionally tokenizer-independent:

    {"id": "...", "prompt_text": "...", "response_text": "...", "source": "..."}

Training code is responsible for applying the model chat template, tokenizing,
and masking prompt tokens with labels=-100. This script only downloads/loads a
source dataset, normalizes common instruction/chat schemas into the contract,
deduplicates, shuffles deterministically, and writes split JSONL files.

Examples
--------
    uv run python scripts/prepare_data.py data=ultrachat_10k
    uv run python scripts/prepare_data.py data=alpaca_50k data.train_size=1000
"""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="prepare_data")
def main(cfg: DictConfig) -> None:
    if cfg.get("hf_cache"):
        hf_home = os.path.expanduser(str(cfg.hf_cache))
        Path(hf_home).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_HUB_CACHE"] = str(Path(hf_home) / "hub")
        os.environ["HF_DATASETS_CACHE"] = str(Path(hf_home) / "datasets")

    _run(cfg)


def _run(cfg: DictConfig) -> None:
    from datasets import load_dataset

    from kdsd.utils.io import write_json, write_jsonl
    from kdsd.utils.logging import get_logger

    log = get_logger("kdsd.prepare_data")
    log.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))
    log.info("HF_HOME=%s", os.environ.get("HF_HOME"))

    data_cfg = cfg.data
    out_dir = _resolve_path(data_cfg.output_dir)
    _prepare_output_dir(out_dir, overwrite=bool(data_cfg.get("overwrite", True)))

    seed = int(cfg.seed)
    train_size = _optional_int(data_cfg.get("train_size"))
    val_size = _optional_int(data_cfg.get("val_size"))
    eval_size = _optional_int(data_cfg.get("eval_size"))

    primary = _load_split(
        load_dataset,
        name=str(data_cfg.name),
        subset=_none_if_empty(data_cfg.get("subset")),
        split=str(data_cfg.split),
    )
    primary_rows = _normalize_dataset(primary, data_cfg, split_name=str(data_cfg.split))
    primary_rows = _dedupe(primary_rows)
    if bool(data_cfg.get("shuffle", True)):
        primary_rows = _shuffle_rows(primary_rows, seed)

    eval_split = _none_if_empty(data_cfg.get("eval_split"))
    if eval_split:
        eval_ds = _load_split(
            load_dataset,
            name=str(data_cfg.name),
            subset=_none_if_empty(data_cfg.get("subset")),
            split=str(eval_split),
        )
        eval_rows_pool = _normalize_dataset(eval_ds, data_cfg, split_name=str(eval_split))
        eval_rows_pool = _dedupe(eval_rows_pool)
        if bool(data_cfg.get("shuffle", True)):
            eval_rows_pool = _shuffle_rows(eval_rows_pool, seed + 1)

        train_rows, remainder = _take(primary_rows, train_size)
        val_rows, _ = _take(remainder, val_size)
        eval_rows, _ = _take(eval_rows_pool, eval_size)
    else:
        train_rows, remainder = _take(primary_rows, train_size)
        val_rows, remainder = _take(remainder, val_size)
        eval_rows, _ = _take(remainder, eval_size)

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)
    write_jsonl(out_dir / "eval.jsonl", eval_rows)
    write_json(out_dir / "meta.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "eval": len(eval_rows),
        },
        "schema": {
            "id": "string",
            "prompt_text": "string",
            "response_text": "string",
            "source": "string",
        },
    })

    log.info(
        "wrote %s: train=%d val=%d eval=%d",
        out_dir,
        len(train_rows),
        len(val_rows),
        len(eval_rows),
    )


def _load_split(load_dataset, *, name: str, subset: str | None, split: str):
    if subset:
        return load_dataset(name, subset, split=split)
    return load_dataset(name, split=split)


def _normalize_dataset(ds, data_cfg: DictConfig, *, split_name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    source = str(data_cfg.get("source") or data_cfg.get("id") or data_cfg.name)
    min_prompt_chars = int(data_cfg.get("min_prompt_chars", 1))
    min_response_chars = int(data_cfg.get("min_response_chars", 1))

    for idx, row in enumerate(ds):
        normalized = _normalize_row(row, idx=idx, source=source, split_name=split_name)
        if normalized is None:
            continue
        if len(normalized["prompt_text"]) < min_prompt_chars:
            continue
        if len(normalized["response_text"]) < min_response_chars:
            continue
        rows.append(normalized)
    return rows


def _normalize_row(
    row: dict[str, Any],
    *,
    idx: int,
    source: str,
    split_name: str,
) -> dict[str, str] | None:
    prompt: str | None = None
    response: str | None = None

    if "prompt_text" in row and "response_text" in row:
        prompt = _clean_text(row.get("prompt_text"))
        response = _clean_text(row.get("response_text"))
    elif "messages" in row:
        prompt, response = _extract_from_messages(row["messages"])
    elif "conversations" in row:
        prompt, response = _extract_from_conversations(row["conversations"])
    elif "instruction" in row and "output" in row:
        prompt = _join_instruction_input(row.get("instruction"), row.get("input"))
        response = _clean_text(row.get("output"))
    elif "prompt" in row and "completion" in row:
        prompt = _clean_text(row.get("prompt"))
        response = _clean_text(row.get("completion"))
    elif "prompt" in row and "response" in row:
        prompt = _clean_text(row.get("prompt"))
        response = _clean_text(row.get("response"))
    elif "question" in row and "answer" in row:
        prompt = _clean_text(row.get("question"))
        response = _clean_text(row.get("answer"))

    if not prompt or not response:
        return None

    raw_id = row.get("id") or row.get("uid")
    sample_id = str(raw_id) if raw_id not in (None, "") else _stable_id(
        source, split_name, idx, prompt, response
    )
    return {
        "id": sample_id,
        "prompt_text": prompt,
        "response_text": response,
        "source": source,
    }


def _extract_from_messages(messages: Any) -> tuple[str | None, str | None]:
    if not isinstance(messages, list):
        return None, None
    user_text: str | None = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).lower()
        content = _clean_text(msg.get("content"))
        if not content:
            continue
        if role == "user" and user_text is None:
            user_text = content
        elif role == "assistant" and user_text is not None:
            return user_text, content
    return None, None


def _extract_from_conversations(conversations: Any) -> tuple[str | None, str | None]:
    if not isinstance(conversations, list):
        return None, None
    user_text: str | None = None
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from") or turn.get("role") or "").lower()
        content = _clean_text(turn.get("value") or turn.get("content"))
        if not content:
            continue
        if role in {"human", "user"} and user_text is None:
            user_text = content
        elif role in {"gpt", "assistant"} and user_text is not None:
            return user_text, content
    return None, None


def _join_instruction_input(instruction: Any, input_text: Any) -> str | None:
    instruction_s = _clean_text(instruction)
    input_s = _clean_text(input_text)
    if instruction_s and input_s:
        return f"{instruction_s}\n\n{input_s}"
    return instruction_s or input_s


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return text or None


def _dedupe(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = hashlib.sha1(
            f"{row['prompt_text']}\n\0\n{row['response_text']}".encode("utf-8")
        ).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _shuffle_rows(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    import random

    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows


def _take(
    rows: list[dict[str, str]],
    n: int | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if n is None or n < 0:
        return rows, []
    n = min(n, len(rows))
    return rows[:n], rows[n:]


def _stable_id(source: str, split_name: str, idx: int, prompt: str, response: str) -> str:
    digest = hashlib.sha1(f"{prompt}\n\0\n{response}".encode("utf-8")).hexdigest()[:12]
    split_slug = split_name.replace("[", "_").replace("]", "").replace(":", "_")
    return f"{source}-{split_slug}-{idx}-{digest}"


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [path / "train.jsonl", path / "val.jsonl", path / "eval.jsonl"]
    if not overwrite and any(p.exists() for p in existing):
        raise FileExistsError(
            f"{path} already contains prepared split files; set data.overwrite=true to replace them"
        )


def _resolve_path(path: str | Path) -> Path:
    p = Path(os.path.expanduser(str(path)))
    return p if p.is_absolute() else _ROOT / p


def _optional_int(value: Any) -> int | None:
    if value in (None, "null", ""):
        return None
    return int(value)


def _none_if_empty(value: Any) -> str | None:
    if value in (None, "null", ""):
        return None
    return str(value)


if __name__ == "__main__":
    main()
