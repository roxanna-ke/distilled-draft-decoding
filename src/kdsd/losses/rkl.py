"""Reverse KL divergence loss: KL(student || teacher).

Also known as "mode-seeking" KL. Minimising this encourages the student to
concentrate its probability mass on a single mode of the teacher, avoiding
spreading mass where the teacher has none.

RKL = sum_x q(x) * log(q(x) / p(x))
    = sum_x q(x) * log q(x) - sum_x q(x) * log p(x)
    = -H(q) + CE(q, p)
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def reverse_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Compute reverse KL divergence KL(q_student || p_teacher) over valid positions.

    Parameters
    ----------
    student_logits : Tensor [B, T, V]
    teacher_logits : Tensor [B, T, V]
    labels : Tensor [B, T]
        -100 on prompt positions to mask.
    temperature : float
        Softmax temperature for both distributions.

    Returns
    -------
    Tensor
        Scalar mean RKL divergence.
    """
    mask = labels != -100  # [B, T]

    if temperature <= 0:
        raise ValueError("temperature must be > 0 for reverse KL")

    # Cast to float32 before softmax for numerical stability (fp16 overflow).
    orig_dtype = student_logits.dtype
    p = F.log_softmax((teacher_logits / temperature).float(), dim=-1)  # teacher log
    q = F.softmax((student_logits / temperature).float(), dim=-1)       # student prob
    q_log = F.log_softmax((student_logits / temperature).float(), dim=-1)  # student log

    # Per-position RKL: sum over vocab of q * (log q - log p)
    per_pos_rkl = (q * (q_log - p)).sum(dim=-1)  # [B, T]

    # Apply mask and average
    n_valid = mask.sum().clamp(min=1)
    return ((per_pos_rkl * mask).sum() / n_valid * (temperature ** 2)).to(orig_dtype)
