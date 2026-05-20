"""Masked cross-entropy loss.

Standard next-token negative log-likelihood with label masking (labels == -100
are ignored). This is both a standalone loss and the hard-target component of
the combined KD loss.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def masked_ce(
    student_logits: Tensor,
    labels: Tensor,
) -> Tensor:
    """Compute mean cross-entropy over non-ignored positions.

    Parameters
    ----------
    student_logits : Tensor [B, T, V]
        Raw logits from the student model.
    labels : Tensor [B, T]
        Target token ids. Positions with value -100 are ignored.

    Returns
    -------
    Tensor
        Scalar mean CE loss over valid positions.
    """
    # Flatten for cross_entropy: [B*T, V] vs [B*T]
    B, T, V = student_logits.shape
    loss = F.cross_entropy(
        student_logits.view(-1, V),
        labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    n_valid = (labels != -100).sum().clamp(min=1)
    return loss / n_valid
