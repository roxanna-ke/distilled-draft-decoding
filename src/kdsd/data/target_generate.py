"""Target response generation for response-source ablations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from kdsd.data.dataset import format_prompt


@torch.inference_mode()
def generate_target_responses(
    records: Iterable[dict[str, Any]],
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_new_tokens: int,
    mode: str = "greedy",
    temperature: float = 0.0,
    top_p: float = 1.0,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    rows = list(records)
    out: list[dict[str, Any]] = []
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(rows), int(batch_size)):
            batch = rows[start:start + int(batch_size)]
            prompts = [format_prompt(tokenizer, str(row["prompt_text"])) for row in batch]
            enc = tokenizer(
                prompts,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            input_len = int(enc["input_ids"].shape[1])
            do_sample = mode == "sampling" and float(temperature) > 0
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = float(temperature)
                gen_kwargs["top_p"] = float(top_p)
            generated = model.generate(**enc, **gen_kwargs)
            new_tokens = generated[:, input_len:]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for row, text in zip(batch, texts):
                next_row = dict(row)
                meta = dict(next_row.get("metadata") or {})
                meta["original_response_text"] = next_row.get("response_text")
                meta["response_generated_at"] = datetime.now(timezone.utc).isoformat()
                next_row["response_text"] = text.strip()
                next_row["source"] = "target"
                next_row["metadata"] = meta
                out.append(next_row)
    finally:
        tokenizer.padding_side = old_padding_side
    return out


def generation_meta(
    *,
    target_model: str,
    seed: int,
    generation_cfg: dict[str, Any],
    source_path: str,
    n_records: int,
) -> dict[str, Any]:
    return {
        "target_model": target_model,
        "seed": int(seed),
        "generation": generation_cfg,
        "source_path": source_path,
        "n_records": int(n_records),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
