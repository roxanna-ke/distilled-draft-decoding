import pytest
import torch
import torch.nn.functional as F

from kdsd.losses import kd_loss
from kdsd.losses.ce import masked_ce
from kdsd.losses.fkl import forward_kl
from kdsd.losses.jsd import jsd_loss
from kdsd.losses.rkl import reverse_kl


def test_masked_ce_ignores_prompt_positions() -> None:
    logits = torch.tensor([
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
    ])
    labels = torch.tensor([[-100, 1, 2]])

    expected = F.cross_entropy(
        logits[:, 1:, :].reshape(-1, 3),
        labels[:, 1:].reshape(-1),
        reduction="mean",
    )

    assert torch.allclose(masked_ce(logits, labels), expected)


@pytest.mark.parametrize("loss_fn", [forward_kl, reverse_kl, jsd_loss])
def test_divergences_are_zero_for_matching_distributions(loss_fn) -> None:
    logits = torch.tensor([
        [[1.5, -0.5, 0.0], [0.1, 0.2, 0.3]],
    ])
    labels = torch.tensor([[-100, 2]])

    assert torch.allclose(
        loss_fn(logits, logits, labels),
        torch.tensor(0.0),
        atol=1e-6,
    )


def test_combined_loss_interpolates_kd_and_ce() -> None:
    student_logits = torch.tensor([
        [[0.0, 1.0, -1.0], [1.0, 0.0, -1.0]],
    ])
    teacher_logits = torch.tensor([
        [[0.0, 1.0, -1.0], [0.2, 1.0, -0.5]],
    ])
    labels = torch.tensor([[-100, 1]])

    out = kd_loss(
        student_logits,
        teacher_logits,
        None,
        None,
        labels,
        kind="fkl",
        temperature=1.0,
        alpha=0.25,
    )

    expected_ce = masked_ce(student_logits, labels)
    expected_kd = forward_kl(student_logits, teacher_logits, labels)
    assert torch.allclose(out["loss"], 0.25 * expected_kd + 0.75 * expected_ce)
    assert torch.allclose(out["ce"], expected_ce)
    assert torch.allclose(out["kd"], expected_kd)
