"""Torch datasets and collators for KD training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from kdsd.utils.io import read_jsonl


class KDDataset(Dataset):
    """Tokenized prompt/response dataset with response-only supervision.

    The text JSONL remains the canonical artifact. This class optionally writes
    a tokenizer-specific cache under `/scratch` so repeated ablations do not pay
    tokenization cost again.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_seq_len: int,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        add_eos: bool = True,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.add_eos = bool(add_eos)
        self.cache_path: Path | None = None

        if use_cache and cache_dir is not None:
            fp = tokenized_cache_fingerprint(
                self.path,
                tokenizer,
                max_seq_len=self.max_seq_len,
                add_eos=self.add_eos,
            )
            self.cache_path = Path(cache_dir) / fp / f"{self.path.stem}.pt"
            if self.cache_path.exists():
                self.examples = torch.load(self.cache_path, weights_only=False)
                return

        rows = read_jsonl(self.path)
        self.examples = [
            tokenize_record(row, tokenizer, max_seq_len=self.max_seq_len, add_eos=self.add_eos)
            for row in rows
        ]
        self.examples = [ex for ex in self.examples if ex["response_mask"].any().item()]

        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.examples, self.cache_path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


class KDCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        self.pad_id = int(pad_id)

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(int(f["input_ids"].shape[0]) for f in features)
        batch: dict[str, list[torch.Tensor]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "response_mask": [],
        }
        for f in features:
            pad = max_len - int(f["input_ids"].shape[0])
            batch["input_ids"].append(_pad_1d(f["input_ids"], pad, self.pad_id))
            batch["attention_mask"].append(_pad_1d(f["attention_mask"], pad, 0))
            batch["labels"].append(_pad_1d(f["labels"], pad, -100))
            batch["response_mask"].append(_pad_1d(f["response_mask"], pad, 0))

        return {
            "input_ids": torch.stack(batch["input_ids"]).long(),
            "attention_mask": torch.stack(batch["attention_mask"]).long(),
            "labels": torch.stack(batch["labels"]).long(),
            "response_mask": torch.stack(batch["response_mask"]).bool(),
        }


def tokenize_record(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_seq_len: int,
    add_eos: bool = True,
) -> dict[str, torch.Tensor]:
    prompt = str(row["prompt_text"])
    response = str(row["response_text"])

    prompt_text = format_prompt(tokenizer, prompt)
    if add_eos and getattr(tokenizer, "eos_token", None):
        response = response + str(tokenizer.eos_token)

    prompt_ids = _encode(tokenizer, prompt_text)
    response_ids = _encode(tokenizer, response)
    input_ids = (prompt_ids + response_ids)[: int(max_seq_len)]
    prompt_len = min(len(prompt_ids), len(input_ids))
    response_len = max(0, len(input_ids) - prompt_len)

    labels = [-100] * prompt_len + input_ids[prompt_len:]
    response_mask = [False] * prompt_len + [True] * response_len

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
    }


def format_prompt(tokenizer: PreTrainedTokenizerBase, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def tokenized_cache_fingerprint(
    path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_seq_len: int,
    add_eos: bool,
) -> str:
    path = Path(path)
    h = hashlib.sha256()
    h.update(_file_sha256(path).encode("utf-8"))
    h.update(str(getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)).encode("utf-8"))
    h.update(str(getattr(tokenizer, "chat_template", "") or "").encode("utf-8"))
    h.update(str(max_seq_len).encode("ascii"))
    h.update(json.dumps({"add_eos": add_eos, "mask": "response_only_v1"}).encode("utf-8"))
    return h.hexdigest()[:16]


def _encode(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    enc = tokenizer(text, add_special_tokens=False)
    ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
    return list(ids)


def _pad_1d(x: torch.Tensor, pad: int, value: int) -> torch.Tensor:
    if pad <= 0:
        return x
    return torch.nn.functional.pad(x, (0, pad), value=value)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
