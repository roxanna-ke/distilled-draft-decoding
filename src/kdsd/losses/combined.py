"""Combined KD loss: alpha * KD_divergence + (1 - alpha) * CE.

This is the single entry point that other modules import. The `kind` parameter
selects which divergence to use for the KD term:

- "ce"  : pure CE (alpha is forced to 0, effectively no KD term)
- "fkl" : Forward KL  KL(teacher || student) — mode-covering
- "rkl" : Reverse KL  KL(student || teacher) — mode-seeking
- "jsd" : Jensen-Shannon Divergence — symmetric

The function also supports a cached top-k teacher path: when
`teacher_logits` is None, `teacher_topk_ids` and `teacher_topk_logp` are
used instead of the full teacher logit tensor. This is currently wired for
FKL (the dominant use-case for top-k caching); RKL and JSD fall back to
requiring full teacher logits.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from kdsd.losses.ce import masked_ce
from kdsd.losses.fkl import forward_kl
from kdsd.losses.jsd import jsd_loss
from kdsd.losses.rkl import reverse_kl


def kd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor | None,
    teacher_topk_ids: Tensor | None,
    teacher_topk_logp: Tensor | None,
    labels: Tensor,
    *,
    kind: Literal["fkl", "rkl", "jsd", "ce"],
    temperature: float = 1.0,
    alpha: float = 0.5,
) -> dict[str, Tensor]:
    """Compute combined KD + CE loss.

    Parameters
    ----------
    student_logits : Tensor [B, T, V]
        Raw logits from the student (draft) model.
    teacher_logits : Tensor | None  [B, T, V]
        Raw logits from the teacher (target) model. None when using the
        cached top-k path.
    teacher_topk_ids : Tensor | None  [B, T, K]
        Top-k token ids from the teacher. Used when teacher_logits is None.
    teacher_topk_logp : Tensor | None  [B, T, K]
        Log-probabilities of the top-k tokens from the teacher.
    labels : Tensor [B, T]
        Target token ids with -100 on prompt (masked) positions.
    kind : {"fkl", "rkl", "jsd", "ce"}
        Which KD divergence to use. "ce" means pure CE (alpha is ignored).
    temperature : float
        Softmax temperature for KD divergences.
    alpha : float
        Weight on the KD term. Final loss = alpha * kd + (1 - alpha) * ce.
        Ignored when kind="ce".

    Returns
    -------
    dict[str, Tensor]
        - "loss": the combined scalar loss
        - "ce":   the CE component (scalar)
        - "kd":   the KD component (scalar; 0.0 when kind="ce")
    """
    # CE is always computed (hard-target supervision)
    ce = masked_ce(student_logits, labels)

    if kind == "ce":
        return {"loss": ce, "ce": ce.detach(), "kd": torch.tensor(0.0, device=ce.device)}

    # Compute the KD divergence
    if teacher_logits is not None:
        # Full teacher logits available — use the standard path
        if kind == "fkl":
            kd = forward_kl(student_logits, teacher_logits, labels, temperature)
        elif kind == "rkl":
            kd = reverse_kl(student_logits, teacher_logits, labels, temperature)
        elif kind == "jsd":
            kd = jsd_loss(student_logits, teacher_logits, labels, temperature)
        else:
            raise ValueError(f"Unknown KD kind: {kind!r}")
    else:
        # Cached top-k path — currently only supported for FKL
        if kind != "fkl":
            raise NotImplementedError(
                f"Cached top-k teacher path is only implemented for FKL, got kind={kind!r}"
            )
        if teacher_topk_ids is None or teacher_topk_logp is None:
            raise ValueError(
                "teacher_topk_ids and teacher_topk_logp are required when teacher_logits=None"
            )
        kd = _forward_kl_topk(
            student_logits, teacher_topk_ids, teacher_topk_logp,
            labels, temperature,
        )

    loss = alpha * kd + (1 - alpha) * ce
    return {"loss": loss, "ce": ce.detach(), "kd": kd.detach()}


def _forward_kl_topk(
    student_logits: Tensor,
    teacher_topk_ids: Tensor,
    teacher_topk_logp: Tensor,
    labels: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """FKL using cached teacher top-k log-probs instead of full logits.

    This avoids materialising the full [B, T, V] teacher logit tensor.
    We approximate FKL ≈ -sum_{x in top-k} p(x) * log q(x) - H(p),
    where H(p) is constant and can be dropped. For the top-k approximation
    to be accurate, K should be large enough (e.g. K=64) that the teacher
    probability mass outside top-k is negligible.

    Parameters
    ----------
    student_logits : Tensor [B, T, V]
    teacher_topk_ids : Tensor [B, T, K]
    teacher_topk_logp : Tensor [B, T, K]
        Teacher log-probs (already temperature-scaled if cached that way).
    labels : Tensor [B, T]
    temperature : float
        Temperature for the student softmax.

    Returns
    -------
    Tensor
        Scalar mean approximate FKL.
    """
    import torch.nn.functional as F

    if temperature <= 0:
        raise ValueError("temperature must be > 0 for top-k forward KL")

    mask = labels != -100  # [B, T]
    if teacher_topk_ids.shape != teacher_topk_logp.shape:
        raise ValueError("teacher_topk_ids and teacher_topk_logp must have the same shape")

    # Student log-probs at teacher's top-k positions
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)  # [B, T, V]
    student_topk_logp = student_log_probs.gather(2, teacher_topk_ids)  # [B, T, K]

    # Teacher probs (exp of logp)
    teacher_topk_p = teacher_topk_logp.exp()  # [B, T, K]

    # Approximate FKL over the cached support. When K covers the full vocab,
    # this is exact; otherwise it drops the teacher tail outside top-k.
    per_pos = (teacher_topk_p * (teacher_topk_logp - student_topk_logp)).sum(dim=-1)

    n_valid = mask.sum().clamp(min=1)
    return (per_pos * mask).sum() / n_valid * (temperature ** 2)
