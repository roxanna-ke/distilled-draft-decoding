"""Aggregation helpers: collapse per-prompt SD stats into the summary numbers."""

from __future__ import annotations

from statistics import mean
from typing import Iterable

from kdsd.sd.instrument import SDStats


def aggregate_sd_stats(stats_list: Iterable[SDStats], gamma: int | None = None) -> dict:
    """Aggregate per-prompt SDStats into eval_summary fields.

    `acceptance_rate` is the fraction of *individual draft proposals* the
    target accepted across all outer steps; defined as
    `sum(accepted_lens) / (n_outer_steps * γ)`. `avg_accepted_tokens` is the
    mean of accepted_lens per outer step.
    """
    stats_list = list(stats_list)
    accepted_lens_all: list[int] = []
    total_new = 0
    target_calls = 0
    draft_calls = 0
    target_s = 0.0
    draft_s = 0.0

    for s in stats_list:
        accepted_lens_all.extend(s.accepted_lens)
        total_new += s.total_new_tokens
        target_calls += s.target_calls
        draft_calls += s.draft_calls
        target_s += s.target_forward_s
        draft_s += s.draft_forward_s

    n_steps = len(accepted_lens_all)
    total_accepted = sum(accepted_lens_all)
    if gamma is not None and n_steps > 0:
        acceptance_rate = float(total_accepted) / float(n_steps * gamma)
    else:
        acceptance_rate = 0.0

    return {
        "acceptance_rate": acceptance_rate,
        "avg_accepted_tokens": float(mean(accepted_lens_all)) if accepted_lens_all else 0.0,
        "total_new_tokens": int(total_new),
        "target_calls": int(target_calls),
        "draft_calls": int(draft_calls),
        "draft_forward_s": float(draft_s),
        "target_forward_s": float(target_s),
        "n_outer_steps": int(n_steps),
    }
