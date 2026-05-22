"""Reverse KL: KL(student || teacher)."""

from __future__ import annotations

import torch


def reverse_kl(student_logp: torch.Tensor, teacher_logp: torch.Tensor) -> torch.Tensor:
    student_p = student_logp.exp()
    return (student_p * (student_logp - teacher_logp)).sum(dim=-1)
