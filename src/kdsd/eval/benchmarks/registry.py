"""Benchmark registry: name → Benchmark class.

Real benchmarks (gpt4 judge, MT-bench, etc.) plug in here later.
For the prototype this is empty; `evaluate_sd.py` therefore writes
`quality_score: {}` in eval_summary.json, which the schema permits.
"""

from __future__ import annotations

from typing import Type

from .base import Benchmark


_REGISTRY: dict[str, Type[Benchmark]] = {}


def register(name: str):
    def deco(cls: Type[Benchmark]):
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get(name: str) -> Type[Benchmark]:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown benchmark {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
