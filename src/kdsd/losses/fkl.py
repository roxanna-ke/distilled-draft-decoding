"""Forward KL: KL(teacher || student)."""

from __future__ import annotations

import torch


def forward_kl(student_logp: torch.Tensor, teacher_logp: torch.Tensor) -> torch.Tensor:
    teacher_p = teacher_logp.exp()
    return (teacher_p * (teacher_logp - student_logp)).sum(dim=-1)
