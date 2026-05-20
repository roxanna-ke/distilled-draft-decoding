"""I/O helpers + eval_summary.json schema validation.

The schema is the one frozen in CLAUDE.md §"Eval contract". `quality_score` is
a dict (possibly empty); aggregate_results.py reads `quality_score.<name>`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


# Required top-level keys in eval_summary.json.
REQUIRED_SUMMARY_KEYS: tuple[str, ...] = (
    "model",
    "target",
    "draft",
    "acceptance_rate",
    "avg_accepted_tokens",
    "vanilla_time_s",
    "sd_time_s",
    "speedup",
    "tokens_per_second",
    "quality_score",
    "decoding",
    "n_prompts",
    "n_warmup",
    "n_repeats",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def validate_eval_summary(summary: dict[str, Any]) -> None:
    """Raise ValueError if `summary` is not a valid eval_summary.json payload."""
    missing = [k for k in REQUIRED_SUMMARY_KEYS if k not in summary]
    if missing:
        raise ValueError(f"eval_summary missing required keys: {missing}")
    if not isinstance(summary["quality_score"], dict):
        raise ValueError("quality_score must be a dict (possibly empty {})")
    if not isinstance(summary["decoding"], dict):
        raise ValueError("decoding must be a dict")
    if "engines" in summary:
        if not isinstance(summary["engines"], dict):
            raise ValueError("engines must be a dict (possibly empty {})")
        for k, v in summary["engines"].items():
            if not isinstance(v, dict):
                raise ValueError(f"engines.{k} must be a dict")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)