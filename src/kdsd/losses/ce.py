"""Masked causal-language-model cross entropy."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute shifted causal LM CE with `-100` ignored."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels.ne(-100)
    if not valid.any():
        return shift_logits.sum() * 0.0
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
