"""Public KD loss entrypoint.

Other modules should import only `kd_loss` from this module. The implementation
uses full online teacher logits in v1; cached top-k teacher logits are left as a
future extension because they bias RKL/JSD comparisons.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from kdsd.losses.ce import masked_ce_loss
from kdsd.losses.fkl import forward_kl
from kdsd.losses.jsd import js_divergence
from kdsd.losses.rkl import reverse_kl


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    teacher_topk_ids: torch.Tensor | None,
    teacher_topk_logp: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    kind: Literal["fkl", "rkl", "jsd", "ce"],
    temperature: float = 1.0,
    alpha: float = 0.5,
    loss_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return `{"loss", "ce", "kd"}` for response-token KD training."""
    kind = kind.lower()
    if kind not in {"fkl", "rkl", "jsd", "ce"}:
        raise ValueError(f"unknown KD loss kind {kind!r}")

    ce = masked_ce_loss(student_logits, labels)
    zero = student_logits.sum() * 0.0
    if kind == "ce":
        return {"loss": ce, "ce": ce.detach(), "kd": zero.detach()}

    if teacher_topk_ids is not None or teacher_topk_logp is not None:
        raise NotImplementedError(
            "Cached top-k teacher logits are intentionally unsupported in v1; "
            "use online teacher logits for unbiased FKL/RKL/JSD comparisons."
        )
    if teacher_logits is None:
        raise ValueError(f"teacher_logits is required for kind={kind!r}")

    kd = _full_distribution_kd(
        student_logits,
        teacher_logits,
        labels,
        kind=kind,
        temperature=float(temperature),
        loss_mask=loss_mask,
    )
    loss = float(alpha) * kd + (1.0 - float(alpha)) * ce
    return {"loss": loss, "ce": ce.detach(), "kd": kd.detach()}


def _full_distribution_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    kind: str,
    temperature: float,
    loss_mask: torch.Tensor | None,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student_logits and teacher_logits must have identical shapes, got "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for KD losses")

    shift_student = student_logits[..., :-1, :].contiguous()
    shift_teacher = teacher_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    if loss_mask is None:
        valid = shift_labels.ne(-100)
    else:
        valid = loss_mask[..., 1:].bool() & shift_labels.ne(-100)

    if not valid.any():
        return shift_student.sum() * 0.0

    s = shift_student[valid] / temperature
    t = shift_teacher[valid] / temperature
    student_logp = F.log_softmax(s.float(), dim=-1)
    teacher_logp = F.log_softmax(t.float(), dim=-1)

    if kind == "fkl":
        per_token = forward_kl(student_logp, teacher_logp)
    elif kind == "rkl":
        per_token = reverse_kl(student_logp, teacher_logp)
    elif kind == "jsd":
        per_token = js_divergence(student_logp, teacher_logp)
    else:  # pragma: no cover - checked by caller
        raise ValueError(kind)
    return per_token.mean().to(student_logits.dtype) * (temperature ** 2)
