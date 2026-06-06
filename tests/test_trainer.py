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


class CacheAwareScriptedLM(nn.Module):
    def __init__(self, scripted_logits):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.scripted_logits = {
            tuple(int(x) for x in key): torch.tensor(value, dtype=torch.float32)
            for key, value in scripted_logits.items()
        }
        self.forward_seq_lens: list[int] = []
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch, seq = input_ids.shape
        assert batch == 1
        self.forward_seq_lens.append(seq)
        prefix = tuple(past_key_values or ())
        key = prefix + tuple(int(x) for x in input_ids[0].tolist())
        vocab = int(next(iter(self.scripted_logits.values())).shape[0])
        logits = self.dummy * torch.zeros((batch, seq, vocab), device=input_ids.device, dtype=torch.float32)
        logits[0, -1, :] = self.scripted_logits[key].to(input_ids.device) + self.dummy
        return SimpleNamespace(logits=logits, past_key_values=key if use_cache else None)


class BatchedTokenLM(nn.Module):
    """Tiny cache-capable LM whose next logits depend only on the input token."""

    def __init__(self, token_logits):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True, pad_token_id=0)
        self.token_logits = torch.tensor(token_logits, dtype=torch.float32)
        self.forward_shapes: list[tuple[int, int]] = []
        self.position_ids_history: list[torch.Tensor | None] = []
        self.cache_position_history: list[torch.Tensor | None] = []
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        cache_position=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        del attention_mask, kwargs
        batch, seq = input_ids.shape
        self.forward_shapes.append((batch, seq))
        self.position_ids_history.append(
            None if position_ids is None else position_ids.detach().cpu().clone()
        )
        self.cache_position_history.append(
            None if cache_position is None else cache_position.detach().cpu().clone()
        )
        logits = self.token_logits.to(input_ids.device)[input_ids] + self.dummy
        cache = past_key_values if past_key_values is not None else ("cache",)
        return SimpleNamespace(logits=logits, past_key_values=cache if use_cache else None)


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


def test_interleaved_training_survives_midrun_eval(tmp_path):
    kwargs = {
        "output_dir": str(tmp_path),
        "max_steps": 1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "learning_rate": 0.0,
        "report_to": [],
        "remove_unused_columns": False,
        "save_strategy": "no",
        "logging_steps": 1,
    }
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    kwargs["eval_steps"] = 1
    if "use_cpu" in params:
        kwargs["use_cpu"] = True
    elif "no_cuda" in params:
        kwargs["no_cuda"] = True

    dataset = [
        {
            "input_ids": torch.tensor([2, 3, 4, 1]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor([-100, -100, 4, 1]),
            "response_mask": torch.tensor([False, False, True, True]),
        }
    ]

    trainer = KDTrainer(
        model=TinyLM(),
        target_model=TinyLM(),
        args=TrainingArguments(**kwargs),
        train_dataset=dataset,
        eval_dataset=dataset,
        kd_cfg={"kind": "fkl", "alpha": 0.5, "temperature": 1.0},
        train_cfg={
            "rollin": "interleaved",
            "interleaved_teacher_topk": 1,
            "interleaved_rollout_tokens": 2,
            "interleaved_student_mode": "greedy",
        },
    )

    result = trainer.train()
    assert result.training_loss >= 0
    eval_log = next(row for row in trainer.state.log_history if "eval_runtime" in row)
    assert eval_log["eval_runtime"] >= 0.0


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
    assert traj.teacher_logits is not None
    assert traj.teacher_logits.shape == (4, 6)
    assert traj.accepted_tokens == 1
    assert traj.proposed_tokens == 2

    student_out, parts, metrics = trainer._compute_interleaved_loss(
        student,
        {
            "input_ids": torch.tensor([[2, 3, 4, 1]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        },
        torch.tensor([[False, False, True, True]]),
    )
    assert student_out is not None
    assert parts["loss"].item() >= 0.0
    assert metrics["acceptance_rate"] == pytest.approx(0.5)
    assert metrics["avg_accepted_tokens"] == pytest.approx(1.0)


def test_interleaved_rollin_uses_kv_cache_for_incremental_steps(tmp_path):
    kwargs = {
        "output_dir": str(tmp_path),
        "max_steps": 1,
        "per_device_train_batch_size": 1,
        "report_to": [],
        "remove_unused_columns": False,
        "save_strategy": "no",
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

    student = CacheAwareScriptedLM(
        {
            (2, 3): [0.0, 0.0, 0.0, 0.0, 9.0, 1.0],
            (2, 3, 5): [0.0, 0.0, 0.0, 8.0, 0.0, 1.0],
            (2, 3, 5, 3): [0.0, 9.0, 0.0, 0.0, 0.0, 1.0],
        }
    )
    teacher = CacheAwareScriptedLM(
        {
            (2, 3): [0.0, 0.0, 0.0, 0.0, 1.0, 9.0],
            (2, 3, 5): [0.0, 0.0, 0.0, 8.0, 0.0, 1.0],
            (2, 3, 5, 3): [0.0, 9.0, 0.0, 0.0, 0.0, 1.0],
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
            "interleaved_rollout_tokens": 3,
            "interleaved_student_mode": "greedy",
        },
    )

    traj = trainer._build_interleaved_example(
        student,
        torch.tensor([2, 3, 4, 1, 1]),
        torch.tensor([False, False, True, True, True]),
        attention_mask=torch.ones(5, dtype=torch.long),
        teacher_topk=1,
        rollout_tokens=3,
    )

    assert traj.input_ids.tolist() == [2, 3, 5, 3, 1]
    assert student.forward_seq_lens == [2, 1, 1]
    assert teacher.forward_seq_lens == [2, 1, 1]


def test_interleaved_loss_reuses_rollin_teacher_logits_without_full_teacher_forward(tmp_path):
    kwargs = {
        "output_dir": str(tmp_path),
        "max_steps": 1,
        "per_device_train_batch_size": 1,
        "report_to": [],
        "remove_unused_columns": False,
        "save_strategy": "no",
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

    student = CacheAwareScriptedLM(
        {
            (2, 3): [0.0, 0.0, 0.0, 0.0, 9.0, 1.0],
            (2, 3, 5): [0.0, 0.0, 0.0, 8.0, 0.0, 1.0],
            (2, 3, 5, 3): [0.0, 9.0, 0.0, 0.0, 0.0, 1.0],
        }
    )
    teacher = CacheAwareScriptedLM(
        {
            (2, 3): [0.0, 0.0, 0.0, 0.0, 1.0, 9.0],
            (2, 3, 5): [0.0, 0.0, 0.0, 8.0, 0.0, 1.0],
            (2, 3, 5, 3): [0.0, 9.0, 0.0, 0.0, 0.0, 1.0],
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

    student_out, parts, metrics = trainer._compute_interleaved_loss(
        student,
        {
            "input_ids": torch.tensor([[2, 3, 4, 1]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        },
        torch.tensor([[False, False, True, True]]),
    )

    assert student_out is not None
    assert parts["loss"].item() >= 0.0
    assert metrics["acceptance_rate"] == pytest.approx(0.5)
    assert student.forward_seq_lens == [2, 1, 4]
    assert teacher.forward_seq_lens == [2, 1]


def test_batched_rollin_gathers_real_prompt_logits_and_keeps_full_batch(tmp_path):
    kwargs = {
        "output_dir": str(tmp_path),
        "max_steps": 1,
        "per_device_train_batch_size": 2,
        "report_to": [],
        "remove_unused_columns": False,
        "save_strategy": "no",
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

    def peaked(token):
        row = [0.0] * 8
        row[token] = 9.0
        return row

    student_rows = [peaked(1) for _ in range(8)]
    teacher_rows = [peaked(1) for _ in range(8)]
    student_rows[3] = peaked(4)
    teacher_rows[3] = peaked(5)
    student_rows[5] = peaked(6)
    teacher_rows[5] = peaked(6)
    student_rows[6] = peaked(7)
    teacher_rows[6] = peaked(7)
    student = BatchedTokenLM(student_rows)
    teacher = BatchedTokenLM(teacher_rows)
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

    batch = trainer._build_interleaved_rollin_batch(
        student,
        {
            "input_ids": torch.tensor([[2, 3, 4, 1], [6, 7, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        },
        torch.tensor(
            [[False, False, True, True], [False, True, False, False]]
        ),
    )

    assert batch.input_ids.tolist() == [[2, 3, 5, 6], [6, 7, 0, 0]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    # The second row is right-padded to prompt width 2; prefill must gather the
    # logit after token 6, not the padded token 0, so the first generated token
    # remains 7.
    assert batch.input_ids[1, 1].item() == 7
    assert batch.accepted_tokens == 2
    assert batch.proposed_tokens == 3
    assert student.forward_shapes == [(2, 2), (2, 1)]
    assert teacher.forward_shapes == [(2, 2), (2, 1)]
    assert student.position_ids_history[1].tolist() == [[2], [1]]
    assert student.cache_position_history[1].tolist() == [2]

    student.forward_shapes.clear()
    teacher.forward_shapes.clear()
    student_out, parts, metrics = trainer._compute_interleaved_loss(
        student,
        {
            "input_ids": torch.tensor([[2, 3, 4, 1], [6, 7, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        },
        torch.tensor(
            [[False, False, True, True], [False, True, False, False]]
        ),
    )
    assert student_out.logits.shape == (2, 4, 8)
    assert parts["loss"].item() >= 0.0
    assert metrics["acceptance_rate"] == pytest.approx(2 / 3)
    assert student.forward_shapes == [(2, 2), (2, 1), (2, 4)]
    assert teacher.forward_shapes == [(2, 2), (2, 1)]


def test_batched_teacher_logits_align_with_next_token_positions(tmp_path):
    kwargs = {
        "output_dir": str(tmp_path),
        "max_steps": 1,
        "per_device_train_batch_size": 2,
        "report_to": [],
        "remove_unused_columns": False,
        "save_strategy": "no",
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

    teacher_rows = torch.arange(64, dtype=torch.float32).reshape(8, 8).tolist()
    student_rows = [[0.0] * 8 for _ in range(8)]
    student_rows[3][4] = 9.0
    teacher_rows[3][5] += 100.0
    student_rows[5][6] = 9.0
    teacher_rows[5][6] += 100.0
    student_rows[6][7] = 9.0
    teacher_rows[6][7] += 100.0
    student = BatchedTokenLM(student_rows)
    teacher = BatchedTokenLM(teacher_rows)
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

    batch = trainer._build_interleaved_rollin_batch(
        student,
        {"input_ids": torch.tensor([[2, 3, 4, 1], [6, 7, 0, 0]])},
        torch.tensor(
            [[False, False, True, True], [False, True, False, False]]
        ),
    )

    assert batch.teacher_logits is not None
    expected = torch.tensor(teacher_rows)
    # Generated token at position t is supervised by teacher logits at t - 1.
    assert torch.equal(batch.teacher_logits[0, 1], expected[3])
    assert torch.equal(batch.teacher_logits[0, 2], expected[5])
    # The shorter right-padded row aligns its single generated token with the
    # prompt-final position, not the padded column.
    assert torch.equal(batch.teacher_logits[1, 0], expected[6])
    assert torch.count_nonzero(batch.teacher_logits[0, 0]) == 0
    assert torch.count_nonzero(batch.teacher_logits[1, 1:]) == 0
