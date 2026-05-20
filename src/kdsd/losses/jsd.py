"""Jensen-Shannon Divergence loss.

A symmetric and smoothed version of KL divergence:
    JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)
where m = (p + q) / 2.

Properties:
- Always non-negative; JSD(p, p) = 0
- Bounded: JSD(p, q) <= log(2)
- Symmetric: JSD(p, q) = JSD(q, p)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def jsd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Compute Jensen-Shannon divergence between teacher and student over valid positions.

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
        Scalar mean JSD.
    """
    mask = labels != -100  # [B, T]

    if temperature <= 0:
        raise ValueError("temperature must be > 0 for JSD")

    p = F.softmax(teacher_logits / temperature, dim=-1)   # teacher
    q = F.softmax(student_logits / temperature, dim=-1)   # student
    m = (p + q) / 2                                       # mixture

    log_p = F.log_softmax(teacher_logits / temperature, dim=-1)
    log_q = F.log_softmax(student_logits / temperature, dim=-1)
    log_m = torch.log(m.clamp_min(1e-12))

    # KL(p || m) = sum p * (log p - log m)
    kl_pm = (p * (log_p - log_m)).sum(dim=-1)  # [B, T]
    # KL(q || m) = sum q * (log q - log m)
    kl_qm = (q * (log_q - log_m)).sum(dim=-1)  # [B, T]

    per_pos_jsd = 0.5 * kl_pm + 0.5 * kl_qm  # [B, T]

    n_valid = mask.sum().clamp(min=1)
    return (per_pos_jsd * mask).sum() / n_valid * (temperature ** 2)
