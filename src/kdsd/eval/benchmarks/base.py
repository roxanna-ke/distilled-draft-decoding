"""Benchmark interface — see README.md §"Benchmark contract"."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class Benchmark(ABC):
    name: str  # key in eval_summary.quality_score

    @abstractmethod
    def score(
        self,
        generations: list[dict],
        target_generations: Optional[list[dict]] = None,
    ) -> float:
        ...
