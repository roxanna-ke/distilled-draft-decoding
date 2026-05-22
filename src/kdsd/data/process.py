"""Dataset normalization for KD training/evaluation splits.

The canonical on-disk format is text JSONL:

    {"id": "...", "prompt_text": "...", "response_text": "...",
     "source": "...", "metadata": {...}}

Tokenization is deliberately a separate, disposable cache layer so text data can
be inspected and reused across tokenizer/preprocessing changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class TextRecord:
    id: str
    prompt_text: str
    response_text: str
    source: str
    metadata: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt_text": self.prompt_text,
            "response_text": self.response_text,
            "source": self.source,
            "metadata": self.metadata,
        }


def normalize_row(
    row: dict[str, Any],
    *,
    family: str,
    index: int,
    dataset_name: str,
    split: str,
) -> TextRecord | None:
    """Normalize a raw dataset row into the project text format.

    The adapters are intentionally permissive because HF instruction datasets
    differ in small schema details across versions. Rows that do not contain a
    usable prompt/response pair return None and are skipped by callers.
    """
    family = family.lower()
    if family == "alpaca":
        prompt, response = _from_alpaca(row)
    else:
        prompt, response = _from_chat_or_prompt_response(row)

    prompt = _clean(prompt)
    response = _clean(response)
    if not prompt or not response:
        return None

    row_id = row.get("id") or row.get("prompt_id") or row.get("conversation_id") or index
    metadata = {
        "dataset": dataset_name,
        "split": split,
        "source_index": index,
    }
    for key in ("prompt_id", "conversation_id", "category", "source"):
        if key in row:
            metadata[key] = row[key]

    return TextRecord(
        id=str(row_id),
        prompt_text=prompt,
        response_text=response,
        source=family,
        metadata=metadata,
    )


def deterministic_split(
    records: list[dict[str, Any]],
    *,
    n_train: int,
    n_val: int,
    n_eval: int,
) -> dict[str, list[dict[str, Any]]]:
    """Slice an already-shuffled list into train/val/eval splits."""
    train_end = max(0, n_train)
    val_end = train_end + max(0, n_val)
    eval_end = val_end + max(0, n_eval)
    return {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "eval": records[val_end:eval_end],
    }


def normalize_rows(
    rows: Iterable[dict[str, Any]],
    *,
    family: str,
    dataset_name: str,
    split: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        rec = normalize_row(
            dict(row),
            family=family,
            index=i,
            dataset_name=dataset_name,
            split=split,
        )
        if rec is not None:
            out.append(rec.to_json())
    return out


def _from_alpaca(row: dict[str, Any]) -> tuple[str | None, str | None]:
    instruction = row.get("instruction") or row.get("prompt")
    inp = row.get("input")
    response = row.get("output") or row.get("response") or row.get("completion")
    if instruction and inp:
        prompt = f"{instruction}\n\nInput:\n{inp}"
    else:
        prompt = instruction
    return prompt, response


def _from_chat_or_prompt_response(row: dict[str, Any]) -> tuple[str | None, str | None]:
    if row.get("prompt_text") and row.get("response_text"):
        return row.get("prompt_text"), row.get("response_text")
    if row.get("prompt") and (row.get("response") or row.get("completion")):
        return row.get("prompt"), row.get("response") or row.get("completion")
    if row.get("instruction") and (row.get("output") or row.get("response")):
        return _from_alpaca(row)

    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, list):
        return _from_messages(messages)

    chosen = row.get("chosen")
    if isinstance(chosen, list):
        return _from_messages(chosen)

    return None, None


def _from_messages(messages: list[Any]) -> tuple[str | None, str | None]:
    user_text: str | None = None
    assistant_text: str | None = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or msg.get("from") or "").lower()
        content = msg.get("content") or msg.get("value")
        if role in {"user", "human"} and user_text is None:
            user_text = content
        elif role in {"assistant", "gpt", "bot"} and user_text is not None:
            assistant_text = content
            break
    return user_text, assistant_text


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
