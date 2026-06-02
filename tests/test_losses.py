import torch
import torch.nn.functional as F

from kdsd.losses import kd_loss


def test_ce_only_matches_shifted_masked_nll():
    logits = torch.tensor(
        [[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0], [1.0, 1.0, 1.0]]]
    )
    labels = torch.tensor([[-100, 1, 2, -100]])

    out = kd_loss(logits, None, None, None, labels, kind="ce")
    expected = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, 3),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(out["loss"], expected)
    assert torch.allclose(out["kd"], torch.tensor(0.0))


def test_fkl_and_rkl_differ_on_asymmetric_distributions():
    student = torch.tensor([[[0.0, 0.0, 0.0], [4.0, 0.0, -2.0], [0.0, 0.0, 0.0]]])
    teacher = torch.tensor([[[0.0, 0.0, 0.0], [-1.0, 3.0, 0.0], [0.0, 0.0, 0.0]]])
    labels = torch.tensor([[-100, -100, 1]])
    mask = labels.ne(-100)

    fkl = kd_loss(student, teacher, None, None, labels, kind="fkl", alpha=1.0, loss_mask=mask)
    rkl = kd_loss(student, teacher, None, None, labels, kind="rkl", alpha=1.0, loss_mask=mask)
    assert not torch.allclose(fkl["loss"], rkl["loss"])


def test_jsd_is_symmetric():
    student = torch.tensor([[[0.0, 0.0], [2.0, -1.0], [0.0, 0.0]]])
    teacher = torch.tensor([[[0.0, 0.0], [-1.0, 2.0], [0.0, 0.0]]])
    labels = torch.tensor([[-100, -100, 1]])

    a = kd_loss(student, teacher, None, None, labels, kind="jsd", alpha=1.0)
    b = kd_loss(teacher, student, None, None, labels, kind="jsd", alpha=1.0)
    assert torch.allclose(a["loss"], b["loss"], atol=1e-6)


def test_response_mask_excludes_prompt_and_pad_tokens():
    student = torch.tensor([[[0.0, 0.0], [3.0, -3.0], [-3.0, 3.0], [0.0, 0.0]]])
    teacher = torch.tensor([[[0.0, 0.0], [-3.0, 3.0], [-3.0, 3.0], [0.0, 0.0]]])
    labels = torch.tensor([[-100, 0, 1, -100]])

    both = kd_loss(student, teacher, None, None, labels, kind="fkl", alpha=1.0)
    response_only = kd_loss(
        student,
        teacher,
        None,
        None,
        labels,
        kind="fkl",
        alpha=1.0,
        loss_mask=torch.tensor([[False, False, True, False]]),
    )
    assert not torch.allclose(response_only["loss"], both["loss"])
    assert response_only["loss"] > both["loss"]


def test_chunked_kd_matches_unchunked_kd():
    torch.manual_seed(0)
    student = torch.randn(2, 5, 11)
    teacher = torch.randn(2, 5, 11)
    labels = torch.tensor([[-100, 2, 3, 4, 5], [-100, -100, 6, 7, 8]])
    mask = labels.ne(-100)

    full = kd_loss(student, teacher, None, None, labels, kind="fkl", alpha=1.0, loss_mask=mask)
    chunked = kd_loss(
        student,
        teacher,
        None,
        None,
        labels,
        kind="fkl",
        alpha=1.0,
        loss_mask=mask,
        chunk_size=2,
    )
    assert torch.allclose(chunked["loss"], full["loss"], atol=1e-6)


def test_cached_topk_path_is_explicitly_unsupported_for_kd():
    logits = torch.zeros(1, 2, 3)
    labels = torch.tensor([[-100, 1]])
    try:
        kd_loss(logits, None, torch.zeros(1), torch.zeros(1), labels, kind="fkl")
    except NotImplementedError as exc:
        assert "online teacher logits" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected NotImplementedError")
