"""Jensen-Shannon divergence between student and teacher distributions."""

from __future__ import annotations

import torch


def js_divergence(student_logp: torch.Tensor, teacher_logp: torch.Tensor) -> torch.Tensor:
    student_p = student_logp.exp()
    teacher_p = teacher_logp.exp()
    mix_p = 0.5 * (student_p + teacher_p)
    mix_logp = mix_p.clamp_min(torch.finfo(mix_p.dtype).tiny).log()
    return 0.5 * (
        (teacher_p * (teacher_logp - mix_logp)).sum(dim=-1)
        + (student_p * (student_logp - mix_logp)).sum(dim=-1)
    )
