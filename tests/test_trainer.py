import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import TrainingArguments

from kdsd.train import KDTrainer


class TinyLM(nn.Module):
    def __init__(self, vocab_size=8, hidden_size=6):
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return SimpleNamespace(logits=self.proj(self.emb(input_ids)))


class ScriptedLM(nn.Module):
    def __init__(self, scripted_logits):
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.scripted_logits = {
            tuple(int(x) for x in key): torch.tensor(value, dtype=torch.float32)
            for key, value in scripted_logits.items()
        }
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch, seq = input_ids.shape
        vocab = int(next(iter(self.scripted_logits.values())).shape[0])
        logits = self.dummy * torch.zeros((batch, seq, vocab), device=input_ids.device, dtype=torch.float32)
        for i in range(batch):
            key = tuple(int(x) for x in input_ids[i].tolist())
            logits[i, -1, :] = self.scripted_logits[key].to(input_ids.device) + self.dummy
        return SimpleNamespace(logits=logits)


def test_kd_trainer_smoke_with_tiny_models(tmp_path):
    dataset = [
        {
            "input_ids": torch.tensor([2, 3, 4, 1]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor([-100, -100, 4, 1]),
            "response_mask": torch.tensor([False, False, True, True]),
        },
        {
            "input_ids": torch.tensor([2, 5, 6, 1]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor([-100, -100, 6, 1]),
            "response_mask": torch.tensor([False, False, True, True]),
        },
    ]

    kwargs = {
        "output_dir": str(tmp_path),
        "max_steps": 1,
        "per_device_train_batch_size": 2,
        "report_to": [],
        "remove_unused_columns": False,
        "save_strategy": "no",
        "logging_steps": 1,
    }
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"
    if "use_cpu" in params:
        kwargs["use_cpu"] = True
    elif "no_cuda" in params:
        kwargs["no_cuda"] = True

    trainer = KDTrainer(
        model=TinyLM(),
        target_model=TinyLM(),
        args=TrainingArguments(**kwargs),
        train_dataset=dataset,
        kd_cfg={"kind": "fkl", "alpha": 0.5, "temperature": 1.0},
    )
    assert trainer.model_accepts_loss_kwargs is False
    result = trainer.train()
    assert result.training_loss >= 0


def test_ce_logging_matches_trainer_loss_with_gradient_accumulation(tmp_path):
    example = {
        "input_ids": torch.tensor([2, 3, 4, 1]),
        "attention_mask": torch.ones(4, dtype=torch.long),
        "labels": torch.tensor([-100, -100, 4, 1]),
        "response_mask": torch.tensor([False, False, True, True]),
    }
    dataset = [example, example]

    kwargs = {
        "output_dir": str(tmp_path),
        "max_steps": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "learning_rate": 0.0,
        "report_to": [],
        "remove_unused_columns": False,
        "save_strategy": "no",
        "logging_steps": 1,
    }
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"
    if "use_cpu" in params:
        kwargs["use_cpu"] = True
    elif "no_cuda" in params:
        kwargs["no_cuda"] = True

    trainer = KDTrainer(
        model=TinyLM(),
        target_model=None,
        args=TrainingArguments(**kwargs),
        train_dataset=dataset,
        kd_cfg={"kind": "ce", "alpha": 0.0, "temperature": 1.0},
    )
    trainer.train()
    train_log = next(row for row in trainer.state.log_history if "loss" in row)
    assert train_log["loss_kd"] == 0.0
    assert train_log["loss"] == pytest.approx(train_log["loss_ce"], abs=1e-4)


def test_interleaved_rollin_replaces_student_tokens_outside_teacher_topk(tmp_path):
    kwargs = {
        "output_dir": str(tmp_path),
        "max_steps": 1,
        "per_device_train_batch_size": 1,
        "report_to": [],
        "remove_unused_columns": False,
        "save_strategy": "no",
        "logging_steps": 1,
    }
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"
    if "use_cpu" in params:
        kwargs["use_cpu"] = True
    elif "no_cuda" in params:
        kwargs["no_cuda"] = True

    student = ScriptedLM(
        {
            (2, 3): [0.0, 0.0, 0.0, 0.0, 9.0, 1.0],
            (2, 3, 5): [0.0, 0.0, 0.0, 8.0, 0.0, 1.0],
            (2, 3, 5, 3): [0.0, 0.0, 0.0, 8.0, 0.0, 1.0],
        }
    )
    teacher = ScriptedLM(
        {
            (2, 3): [0.0, 0.0, 0.0, 0.0, 1.0, 9.0],
            (2, 3, 5): [0.0, 0.0, 0.0, 8.0, 0.0, 1.0],
            (2, 3, 5, 3): [0.0, 0.0, 0.0, 8.0, 0.0, 1.0],
        }
    )
    trainer = KDTrainer(
        model=student,
        target_model=teacher,
        args=TrainingArguments(**kwargs),
        train_dataset=[],
        kd_cfg={"kind": "fkl", "alpha": 1.0, "temperature": 1.0},
        train_cfg={
            "rollin": "interleaved",
            "interleaved_teacher_topk": 1,
            "interleaved_rollout_tokens": 2,
            "interleaved_student_mode": "greedy",
        },
    )

    token, accepted = trainer._select_interleaved_token(
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 9.0, 1.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 9.0]]),
        1,
    )
    assert token == 5
    assert accepted is False

    traj = trainer._build_interleaved_example(
        student,
        torch.tensor([2, 3, 4, 1]),
        torch.tensor([False, False, True, True]),
        attention_mask=torch.ones(4, dtype=torch.long),
        teacher_topk=1,
        rollout_tokens=2,
    )
    assert traj.input_ids.tolist() == [2, 3, 5, 3]
    assert traj.labels.tolist() == [-100, -100, 5, 3]
    assert traj.accepted_tokens == 1
    assert traj.proposed_tokens == 2

    parts, metrics = trainer._compute_interleaved_loss(
        student,
        {
            "input_ids": torch.tensor([[2, 3, 4, 1]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        },
        torch.tensor([[False, False, True, True]]),
    )
    assert parts["loss"].item() >= 0.0
    assert metrics["acceptance_rate"] == pytest.approx(0.5)
    assert metrics["avg_accepted_tokens"] == pytest.approx(1.0)
