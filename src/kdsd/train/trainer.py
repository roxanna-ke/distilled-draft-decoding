"""HF Trainer subclass for online target-forward KD."""

from __future__ import annotations

from typing import Any

import torch
from transformers import Trainer

from kdsd.losses import kd_loss


class KDTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        target_model: torch.nn.Module | None,
        kd_cfg: dict,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kd_cfg = dict(kd_cfg)
        self.target = target_model
        if self.target is not None:
            self.target = self.target.eval().requires_grad_(False)
        # Qwen forwards accept **kwargs, so HF Trainer assumes the model/loss
        # handles num_items_in_batch normalization itself. Our custom loss is
        # already a per-token mean, so keep Trainer's standard GA scaling.
        self.model_accepts_loss_kwargs = False
        self._loss_part_sums: dict[str, float] = {"loss_ce": 0.0, "loss_kd": 0.0}
        self._loss_part_count = 0

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ):
        labels = inputs.pop("labels")
        response_mask = inputs.pop("response_mask", labels.ne(-100))

        student_out = model(**inputs)
        teacher_logits = None
        if self.kd_cfg["kind"] != "ce":
            if self.target is None:
                raise ValueError("target_model is required for KD losses")
            with torch.no_grad():
                teacher_logits = self.target(**inputs).logits

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
            chunk_size=self.kd_cfg.get("chunk_size"),
        )
        if model.training:
            self._loss_part_sums["loss_ce"] += float(loss_parts["ce"].detach().cpu())
            self._loss_part_sums["loss_kd"] += float(loss_parts["kd"].detach().cpu())
            self._loss_part_count += 1
        if return_outputs:
            return loss_parts["loss"], student_out
        return loss_parts["loss"]

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
        super().log(logs, *args, **kwargs)
