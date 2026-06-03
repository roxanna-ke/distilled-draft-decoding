"""HF Trainer subclass for online target-forward KD."""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import torch
from transformers import Trainer

from kdsd.losses import kd_loss


@dataclass
class InterleavedTrajectory:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    response_mask: torch.Tensor
    accepted_tokens: int
    proposed_tokens: int


class KDTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        target_model: torch.nn.Module | None,
        kd_cfg: dict,
        train_cfg: dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kd_cfg = dict(kd_cfg)
        self.train_cfg = dict(train_cfg or {})
        self.target = target_model
        if self.target is not None:
            self.target = self.target.eval().requires_grad_(False)
        # Qwen forwards accept **kwargs, so HF Trainer assumes the model/loss
        # handles num_items_in_batch normalization itself. Our custom loss is
        # already a per-token mean, so keep Trainer's standard GA scaling.
        self.model_accepts_loss_kwargs = False
        self._loss_part_sums: dict[str, float] = {"loss_ce": 0.0, "loss_kd": 0.0}
        self._loss_part_count = 0
        self._rollin_metric_sums: dict[str, float] = {
            "rollin_acceptance_rate": 0.0,
            "rollin_avg_accepted_tokens": 0.0,
        }
        self._rollin_metric_count = 0

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ):
        labels = inputs["labels"]
        response_mask = inputs.get("response_mask", labels.ne(-100))

        if self._use_interleaved_rollin():
            loss_parts, rollin_metrics = self._compute_interleaved_loss(model, inputs, response_mask)
            student_out = None
        else:
            student_out, loss_parts = self._compute_full_sequence_loss(
                model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=labels,
                response_mask=response_mask,
            )
            rollin_metrics = None
        if model.training:
            self._loss_part_sums["loss_ce"] += float(loss_parts["ce"].detach().cpu())
            self._loss_part_sums["loss_kd"] += float(loss_parts["kd"].detach().cpu())
            self._loss_part_count += 1
            if rollin_metrics is not None:
                self._rollin_metric_sums["rollin_acceptance_rate"] += float(rollin_metrics["acceptance_rate"])
                self._rollin_metric_sums["rollin_avg_accepted_tokens"] += float(rollin_metrics["avg_accepted_tokens"])
                self._rollin_metric_count += 1
        if return_outputs:
            return loss_parts["loss"], student_out
        return loss_parts["loss"]

    def _use_interleaved_rollin(self) -> bool:
        return str(self.train_cfg.get("rollin", "teacher_forcing")).lower() == "interleaved"

    def _compute_full_sequence_loss(
        self,
        model: torch.nn.Module,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        student_out = model(**model_inputs)
        teacher_logits = None
        if self.kd_cfg["kind"] != "ce":
            if self.target is None:
                raise ValueError("target_model is required for KD losses")
            with torch.no_grad():
                teacher_logits = self.target(**model_inputs).logits

        loss_parts = kd_loss(
            student_out.logits,
            teacher_logits,
            None,
            None,
            labels,
            kind=self.kd_cfg["kind"],
            temperature=float(self.kd_cfg.get("temperature", 1.0)),
            alpha=float(self.kd_cfg.get("alpha", 0.5)),
            loss_mask=response_mask,
        )
        return student_out, loss_parts

    def _compute_interleaved_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        response_mask: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        if self.kd_cfg["kind"] != "ce" and self.target is None:
            raise ValueError("target_model is required for interleaved KD losses")

        trajectories = self._build_interleaved_batch(model, inputs, response_mask)
        batch_losses: list[dict[str, torch.Tensor]] = []
        accepted_tokens = 0
        proposed_tokens = 0
        for traj in trajectories:
            _, parts = self._compute_full_sequence_loss(
                model,
                input_ids=traj.input_ids.unsqueeze(0),
                attention_mask=traj.attention_mask.unsqueeze(0),
                labels=traj.labels.unsqueeze(0),
                response_mask=traj.response_mask.unsqueeze(0),
            )
            batch_losses.append(parts)
            accepted_tokens += traj.accepted_tokens
            proposed_tokens += traj.proposed_tokens

        losses = {
            key: torch.stack([parts[key] for parts in batch_losses]).mean()
            for key in ("loss", "ce", "kd")
        }
        metrics = {
            "acceptance_rate": (accepted_tokens / proposed_tokens) if proposed_tokens > 0 else 0.0,
            "avg_accepted_tokens": accepted_tokens / max(len(trajectories), 1),
        }
        return losses, metrics

    def _build_interleaved_batch(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        response_mask: torch.Tensor,
    ) -> list[InterleavedTrajectory]:
        teacher_topk = int(self.train_cfg.get("interleaved_teacher_topk", 1))
        rollout_tokens = int(self.train_cfg.get("interleaved_rollout_tokens", 64))
        if teacher_topk <= 0:
            raise ValueError("train.interleaved_teacher_topk must be >= 1")
        if rollout_tokens <= 0:
            raise ValueError("train.interleaved_rollout_tokens must be >= 1")

        trajectories: list[InterleavedTrajectory] = []
        batch_input_ids = inputs["input_ids"]
        batch_attention_mask = inputs.get("attention_mask")
        for idx in range(batch_input_ids.shape[0]):
            attention_mask = None if batch_attention_mask is None else batch_attention_mask[idx]
            trajectory = self._build_interleaved_example(
                model,
                batch_input_ids[idx],
                response_mask[idx],
                attention_mask=attention_mask,
                teacher_topk=teacher_topk,
                rollout_tokens=rollout_tokens,
            )
            trajectories.append(trajectory)
        return trajectories

    def _build_interleaved_example(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        response_mask: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        teacher_topk: int,
        rollout_tokens: int,
    ) -> InterleavedTrajectory:
        response_positions = response_mask.bool().nonzero(as_tuple=False).flatten()
        if response_positions.numel() == 0:
            raise ValueError("interleaved rollin requires both prompt and response tokens")
        prompt_len = int(response_positions[0].item())
        response_len = min(int(response_positions.numel()), rollout_tokens)
        if prompt_len <= 0:
            raise ValueError("interleaved rollin requires both prompt and response tokens")

        current_ids = input_ids[:prompt_len].unsqueeze(0)
        if attention_mask is None:
            current_attention_mask = torch.ones_like(current_ids)
        else:
            current_attention_mask = attention_mask[:prompt_len].unsqueeze(0)

        mixed_tokens: list[int] = []
        accepted_tokens = 0

        for _ in range(response_len):
            student_out = model(input_ids=current_ids, attention_mask=current_attention_mask)
            student_next = student_out.logits[:, -1, :]

            teacher_next = None
            if self.target is not None:
                with torch.no_grad():
                    teacher_out = self.target(input_ids=current_ids, attention_mask=current_attention_mask)
                    teacher_next = teacher_out.logits[:, -1, :]

            next_token, accepted = self._select_interleaved_token(
                student_next,
                teacher_next,
                teacher_topk,
            )
            mixed_tokens.append(next_token)
            accepted_tokens += int(accepted)

            next_token_tensor = torch.tensor([[next_token]], device=current_ids.device, dtype=current_ids.dtype)
            current_ids = torch.cat([current_ids, next_token_tensor], dim=1)
            next_attention = torch.ones((1, 1), device=current_ids.device, dtype=current_attention_mask.dtype)
            current_attention_mask = torch.cat([current_attention_mask, next_attention], dim=1)

        mixed_input_ids = current_ids.squeeze(0)
        mixed_attention_mask = current_attention_mask.squeeze(0)
        labels = mixed_input_ids.new_full(mixed_input_ids.shape, -100)
        labels[prompt_len:] = mixed_input_ids[prompt_len:]
        mixed_response_mask = torch.zeros_like(mixed_input_ids, dtype=torch.bool)
        mixed_response_mask[prompt_len:] = True
        return InterleavedTrajectory(
            input_ids=mixed_input_ids,
            attention_mask=mixed_attention_mask,
            labels=labels,
            response_mask=mixed_response_mask,
            accepted_tokens=accepted_tokens,
            proposed_tokens=response_len,
        )

    def _propose_student_token(self, student_next: torch.Tensor) -> int:
        mode = str(self.train_cfg.get("interleaved_student_mode", "greedy")).lower()
        if mode == "greedy":
            return int(student_next.argmax(dim=-1).item())
        if mode != "sample":
            raise ValueError(f"Unsupported train.interleaved_student_mode={mode!r}")

        temperature = float(self.train_cfg.get("interleaved_student_temperature", 0.3))
        top_p = float(self.train_cfg.get("interleaved_student_top_p", 1.0))
        if temperature <= 0:
            raise ValueError("train.interleaved_student_temperature must be > 0 for sampling")
        logits = (student_next / temperature).squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep = cumulative <= top_p
            keep[0] = True
            filtered = torch.zeros_like(probs)
            filtered.scatter_(0, sorted_idx[keep], sorted_probs[keep])
            probs = filtered / filtered.sum()
        return int(torch.multinomial(probs, num_samples=1).item())

    def _select_interleaved_token(
        self,
        student_next: torch.Tensor,
        teacher_next: torch.Tensor | None,
        teacher_topk: int,
    ) -> tuple[int, bool]:
        student_token = self._propose_student_token(student_next)
        if teacher_next is None:
            return student_token, True
        topk = min(teacher_topk, int(teacher_next.shape[-1]))
        teacher_topk_ids = teacher_next.topk(k=topk, dim=-1).indices
        if bool((teacher_topk_ids == student_token).any().item()):
            return student_token, True
        return int(teacher_next.argmax(dim=-1).item()), False

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if "loss" in logs and self._loss_part_count > 0:
            logs = {
                **logs,
                **{
                    k: v / self._loss_part_count
                    for k, v in self._loss_part_sums.items()
                },
            }
            self._loss_part_sums = {"loss_ce": 0.0, "loss_kd": 0.0}
            self._loss_part_count = 0
        if "loss" in logs and self._rollin_metric_count > 0:
            logs = {
                **logs,
                **{
                    k: v / self._rollin_metric_count
                    for k, v in self._rollin_metric_sums.items()
                },
            }
            self._rollin_metric_sums = {
                "rollin_acceptance_rate": 0.0,
                "rollin_avg_accepted_tokens": 0.0,
            }
            self._rollin_metric_count = 0
        super().log(logs, *args, **kwargs)
