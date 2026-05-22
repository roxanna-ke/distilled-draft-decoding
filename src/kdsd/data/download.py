"""Thin wrappers around `datasets.load_dataset`.

Keeping this indirection makes scripts easier to test and keeps all HF dataset
access in the data package.
"""

from __future__ import annotations

from typing import Any


def load_hf_split(dataset_cfg: dict[str, Any]):
    from datasets import load_dataset

    name = dataset_cfg["name"]
    split = dataset_cfg.get("split", "train")
    kwargs = {k: v for k, v in dataset_cfg.items() if k not in {"name", "split"}}
    return load_dataset(name, split=split, **kwargs)
