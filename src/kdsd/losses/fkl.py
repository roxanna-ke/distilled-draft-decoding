"""Forward KL divergence loss: KL(teacher || student).

Also known as "mode-covering" KL. Minimising this encourages the student to
spread probability mass to cover all modes of the teacher distribution.

FKL = sum_x p(x) * log(p(x) / q(x))
    = sum_x p(x) * log p(x) - sum_x p(x) * log q(x)
    = -H(p) + CE(p, q)

Since H(p) is constant w.r.t. the student, minimising FKL is equivalent to
minimising the cross-entropy CE(p, q) = -sum_x p(x) * log q(x).
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def forward_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Compute forward KL divergence KL(p_teacher || q_student) over valid positions.

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
        Scalar mean FK divergence.
    """
    # Build a mask that is True for valid (response) positions
    mask = labels != -100  # [B, T]

    if temperature <= 0:
        raise ValueError("temperature must be > 0 for forward KL")

    # Soften distributions with temperature.
    # Cast to float32 before softmax for numerical stability (fp16 overflow).
    orig_dtype = student_logits.dtype
    log_p = F.log_softmax((teacher_logits / temperature).float(), dim=-1)
    log_q = F.log_softmax((student_logits / temperature).float(), dim=-1)
    p = log_p.exp()

    # Per-position KL: sum over vocab of p * (log p - log q).
    per_pos_kl = (p * (log_p - log_q)).sum(dim=-1)  # [B, T]

    # Apply mask and average
    n_valid = mask.sum().clamp(min=1)
    return ((per_pos_kl * mask).sum() / n_valid * (temperature ** 2)).to(orig_dtype)
