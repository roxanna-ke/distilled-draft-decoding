"""HF Trainer subclass for online target-forward KD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import Trainer

from kdsd.losses import kd_loss


@dataclass
class InterleavedTrajectory:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    response_mask: torch.Tensor
    teacher_logits: torch.Tensor | None
    accepted_tokens: int
    proposed_tokens: int


@dataclass
class BatchedInterleavedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    response_mask: torch.Tensor
    teacher_logits: torch.Tensor | None
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
        del kwargs
        labels = inputs["labels"]
        response_mask = inputs.get("response_mask", labels.ne(-100))

        if self._use_interleaved_rollin():
            student_out, loss_parts, rollin_metrics = self._compute_interleaved_loss(
                model, inputs, response_mask
            )
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
                self._rollin_metric_sums["rollin_acceptance_rate"] += float(
                    rollin_metrics["acceptance_rate"]
                )
                self._rollin_metric_sums["rollin_avg_accepted_tokens"] += float(
                    rollin_metrics["avg_accepted_tokens"]
                )
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
        teacher_logits: torch.Tensor | None = None,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        student_out = model(**model_inputs)
        if self.kd_cfg["kind"] != "ce":
            if self.target is None:
                raise ValueError("target_model is required for KD losses")
            if teacher_logits is None:
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
    ) -> tuple[Any, dict[str, torch.Tensor], dict[str, float]]:
        if self.kd_cfg["kind"] != "ce" and self.target is None:
            raise ValueError("target_model is required for interleaved KD losses")

        batch = self._build_interleaved_rollin_batch(model, inputs, response_mask)
        student_out, loss_parts = self._compute_full_sequence_loss(
            model,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            response_mask=batch.response_mask,
            teacher_logits=batch.teacher_logits,
        )
        batch_size = int(batch.input_ids.shape[0])
        metrics = {
            "acceptance_rate": (
                batch.accepted_tokens / batch.proposed_tokens
                if batch.proposed_tokens > 0
                else 0.0
            ),
            "avg_accepted_tokens": batch.accepted_tokens / max(batch_size, 1),
        }
        return student_out, loss_parts, metrics

    def _build_interleaved_rollin_batch(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        response_mask: torch.Tensor,
    ) -> BatchedInterleavedBatch:
        teacher_topk = int(self.train_cfg.get("interleaved_teacher_topk", 1))
        rollout_tokens = int(self.train_cfg.get("interleaved_rollout_tokens", 32))
        if teacher_topk <= 0:
            raise ValueError("train.interleaved_teacher_topk must be >= 1")
        if rollout_tokens <= 0:
            raise ValueError("train.interleaved_rollout_tokens must be >= 1")

        input_ids = inputs["input_ids"]
        batch_size = int(input_ids.shape[0])
        device = input_ids.device
        prompt_lens = self._prompt_lengths(response_mask)
        response_lens = response_mask.long().sum(dim=1)
        target_lens = response_lens.clamp_max(rollout_tokens)
        max_prompt_len = int(prompt_lens.max().item())
        max_rollout_len = int(target_lens.max().item())
        pad_token_id = self._pad_token_id(model)

        # v1 uses a fixed right-padded physical cache layout. Prompt padding is
        # masked, while explicit logical position_ids preserve Qwen RoPE positions.
        prompt_ids = input_ids.new_full((batch_size, max_prompt_len), pad_token_id)
        prompt_attention = input_ids.new_zeros((batch_size, max_prompt_len))
        for row, prompt_len in enumerate(prompt_lens.tolist()):
            prompt_ids[row, :prompt_len] = input_ids[row, :prompt_len]
            prompt_attention[row, :prompt_len] = 1

        generated = input_ids.new_full((batch_size, max_rollout_len), pad_token_id)
        generated_counts = input_ids.new_zeros(batch_size)
        accepted_counts = input_ids.new_zeros(batch_size)
        teacher_steps: torch.Tensor | None = None

        student_was_training = model.training
        if student_was_training:
            model.eval()
        try:
            with torch.no_grad():
                student_next, student_cache = self._rollin_prefill(
                    model, prompt_ids, prompt_attention, prompt_lens
                )
                teacher_next = None
                teacher_cache = None
                if self.target is not None:
                    teacher_next, teacher_cache = self._rollin_prefill(
                        self.target, prompt_ids, prompt_attention, prompt_lens
                    )

                current_attention = prompt_attention
                for step in range(max_rollout_len):
                    # Keep [B, 1] for every cache step. Finished rows receive a
                    # masked dummy token and are ignored by active_mask.
                    active_mask = generated_counts.lt(target_lens)
                    if teacher_next is not None:
                        if teacher_steps is None:
                            teacher_steps = teacher_next.new_zeros(
                                (batch_size, max_rollout_len, teacher_next.shape[-1])
                            )
                        teacher_steps[active_mask, step] = teacher_next[active_mask]

                    next_tokens, accepted = self._select_interleaved_tokens(
                        student_next,
                        teacher_next,
                        teacher_topk,
                        active_mask,
                        pad_token_id=pad_token_id,
                    )
                    generated[:, step] = next_tokens
                    accepted_counts += accepted.long()
                    generated_counts += active_mask.long()

                    if step + 1 >= max_rollout_len:
                        break
                    next_attention = active_mask.to(current_attention.dtype).unsqueeze(1)
                    current_attention = torch.cat([current_attention, next_attention], dim=1)
                    logical_positions = (prompt_lens + step).unsqueeze(1)
                    cache_position = torch.tensor(
                        [max_prompt_len + step], device=device, dtype=torch.long
                    )
                    student_next, student_cache = self._rollin_advance(
                        model,
                        next_tokens.unsqueeze(1),
                        current_attention,
                        logical_positions,
                        cache_position,
                        student_cache,
                        full_input_ids=torch.cat([prompt_ids, generated[:, : step + 1]], dim=1),
                    )
                    if self.target is not None:
                        teacher_next, teacher_cache = self._rollin_advance(
                            self.target,
                            next_tokens.unsqueeze(1),
                            current_attention,
                            logical_positions,
                            cache_position,
                            teacher_cache,
                            full_input_ids=torch.cat(
                                [prompt_ids, generated[:, : step + 1]], dim=1
                            ),
                        )
        finally:
            if student_was_training:
                model.train()

        return self._assemble_interleaved_batch(
            prompt_ids=prompt_ids,
            prompt_lens=prompt_lens,
            generated=generated,
            generated_lens=target_lens,
            teacher_steps=teacher_steps,
            accepted_tokens=int(accepted_counts.sum().item()),
            proposed_tokens=int(target_lens.sum().item()),
            pad_token_id=pad_token_id,
        )

    def _assemble_interleaved_batch(
        self,
        *,
        prompt_ids: torch.Tensor,
        prompt_lens: torch.Tensor,
        generated: torch.Tensor,
        generated_lens: torch.Tensor,
        teacher_steps: torch.Tensor | None,
        accepted_tokens: int,
        proposed_tokens: int,
        pad_token_id: int,
    ) -> BatchedInterleavedBatch:
        batch_size = int(prompt_ids.shape[0])
        final_lens = prompt_lens + generated_lens
        max_final_len = int(final_lens.max().item())
        mixed_ids = prompt_ids.new_full((batch_size, max_final_len), pad_token_id)
        attention_mask = prompt_ids.new_zeros((batch_size, max_final_len))
        labels = prompt_ids.new_full((batch_size, max_final_len), -100)
        response_mask = torch.zeros(
            (batch_size, max_final_len), device=prompt_ids.device, dtype=torch.bool
        )
        teacher_logits = None
        if teacher_steps is not None:
            teacher_logits = teacher_steps.new_zeros(
                (batch_size, max_final_len, teacher_steps.shape[-1])
            )

        for row in range(batch_size):
            prompt_len = int(prompt_lens[row].item())
            generated_len = int(generated_lens[row].item())
            final_len = prompt_len + generated_len
            mixed_ids[row, :prompt_len] = prompt_ids[row, :prompt_len]
            mixed_ids[row, prompt_len:final_len] = generated[row, :generated_len]
            attention_mask[row, :final_len] = 1
            labels[row, prompt_len:final_len] = generated[row, :generated_len]
            response_mask[row, prompt_len:final_len] = True
            if teacher_logits is not None:
                start = prompt_len - 1
                teacher_logits[row, start : start + generated_len] = teacher_steps[
                    row, :generated_len
                ]

        return BatchedInterleavedBatch(
            input_ids=mixed_ids,
            attention_mask=attention_mask,
            labels=labels,
            response_mask=response_mask,
            teacher_logits=teacher_logits,
            accepted_tokens=accepted_tokens,
            proposed_tokens=proposed_tokens,
        )

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
        old_topk = self.train_cfg.get("interleaved_teacher_topk")
        old_rollout = self.train_cfg.get("interleaved_rollout_tokens")
        self.train_cfg["interleaved_teacher_topk"] = teacher_topk
        self.train_cfg["interleaved_rollout_tokens"] = rollout_tokens
        try:
            batch = self._build_interleaved_rollin_batch(
                model,
                {
                    "input_ids": input_ids.unsqueeze(0),
                    "attention_mask": (
                        attention_mask.unsqueeze(0) if attention_mask is not None else None
                    ),
                },
                response_mask.unsqueeze(0),
            )
        finally:
            if old_topk is None:
                self.train_cfg.pop("interleaved_teacher_topk", None)
            else:
                self.train_cfg["interleaved_teacher_topk"] = old_topk
            if old_rollout is None:
                self.train_cfg.pop("interleaved_rollout_tokens", None)
            else:
                self.train_cfg["interleaved_rollout_tokens"] = old_rollout
        final_len = int(batch.attention_mask[0].sum().item())
        return InterleavedTrajectory(
            input_ids=batch.input_ids[0, :final_len],
            attention_mask=batch.attention_mask[0, :final_len],
            labels=batch.labels[0, :final_len],
            response_mask=batch.response_mask[0, :final_len],
            teacher_logits=(
                batch.teacher_logits[0, :final_len]
                if batch.teacher_logits is not None
                else None
            ),
            accepted_tokens=batch.accepted_tokens,
            proposed_tokens=batch.proposed_tokens,
        )

    def _rollin_prefill(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).expand(batch_size, -1)
        cache_position = torch.arange(seq_len, device=input_ids.device)
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
        )
        gather_rows = torch.arange(batch_size, device=input_ids.device)
        next_logits = out.logits[gather_rows, prompt_lens - 1]
        return next_logits, getattr(out, "past_key_values", None)

    def _rollin_advance(
        self,
        model: torch.nn.Module,
        next_token: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        cache: Any,
        *,
        full_input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        if cache is None:
            last_real = self._last_true_positions(attention_mask)
            return self._rollin_prefill(model, full_input_ids, attention_mask, last_real + 1)
        out = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        )
        return out.logits[:, -1, :], getattr(out, "past_key_values", cache)

    def _propose_student_tokens(self, student_next: torch.Tensor) -> torch.Tensor:
        mode = str(self.train_cfg.get("interleaved_student_mode", "greedy")).lower()
        if mode == "greedy":
            return student_next.argmax(dim=-1)
        if mode != "sample":
            raise ValueError(f"Unsupported train.interleaved_student_mode={mode!r}")

        temperature = float(self.train_cfg.get("interleaved_student_temperature", 0.3))
        top_p = float(self.train_cfg.get("interleaved_student_top_p", 1.0))
        if temperature <= 0:
            raise ValueError("train.interleaved_student_temperature must be > 0 for sampling")
        probs = torch.softmax(student_next.float() / temperature, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep = cumulative <= top_p
            keep[:, 0] = True
            filtered = torch.zeros_like(probs)
            filtered.scatter_(1, sorted_idx, sorted_probs * keep)
            probs = filtered / filtered.sum(dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1).squeeze(1)

    def _select_interleaved_tokens(
        self,
        student_next: torch.Tensor,
        teacher_next: torch.Tensor | None,
        teacher_topk: int,
        active_mask: torch.Tensor,
        *,
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        student_tokens = self._propose_student_tokens(student_next)
        if teacher_next is None:
            accepted = active_mask.clone()
            chosen = student_tokens
        else:
            topk = min(teacher_topk, int(teacher_next.shape[-1]))
            teacher_topk_ids = teacher_next.topk(k=topk, dim=-1).indices
            accepted = teacher_topk_ids.eq(student_tokens.unsqueeze(1)).any(dim=1)
            accepted &= active_mask
            teacher_tokens = teacher_next.argmax(dim=-1)
            chosen = torch.where(accepted, student_tokens, teacher_tokens)
        chosen = torch.where(
            active_mask,
            chosen,
            torch.full_like(chosen, pad_token_id),
        )
        return chosen.long(), accepted

    def _propose_student_token(self, student_next: torch.Tensor) -> int:
        return int(self._propose_student_tokens(student_next).item())

    def _select_interleaved_token(
        self,
        student_next: torch.Tensor,
        teacher_next: torch.Tensor | None,
        teacher_topk: int,
    ) -> tuple[int, bool]:
        chosen, accepted = self._select_interleaved_tokens(
            student_next,
            teacher_next,
            teacher_topk,
            torch.ones(student_next.shape[0], device=student_next.device, dtype=torch.bool),
            pad_token_id=0,
        )
        return int(chosen.item()), bool(accepted.item())

    @staticmethod
    def _prompt_lengths(response_mask: torch.Tensor) -> torch.Tensor:
        has_response = response_mask.any(dim=1)
        if not bool(has_response.all().item()):
            raise ValueError("interleaved rollin requires response tokens for every sample")
        prompt_lens = response_mask.long().argmax(dim=1)
        if bool(prompt_lens.le(0).any().item()):
            raise ValueError("interleaved rollin requires both prompt and response tokens")
        return prompt_lens

    @staticmethod
    def _last_true_positions(attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
        masked = positions.unsqueeze(0).expand_as(attention_mask).masked_fill(
            attention_mask.eq(0), -1
        )
        return masked.max(dim=1).values

    @staticmethod
    def _pad_token_id(model: torch.nn.Module) -> int:
        config = getattr(model, "config", None)
        pad_token_id = getattr(config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(config, "eos_token_id", None)
        return int(pad_token_id or 0)

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if "loss" in logs and self._loss_part_count > 0:
            logs = {
                **logs,
                **{k: v / self._loss_part_count for k, v in self._loss_part_sums.items()},
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
