"""Minimal Accelerate training loop. Paper-specific training policies are extension points."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from onemodel.config import EvaluationConfig, TrainingConfig
from onemodel.data.schemas import Batch
from onemodel.models.outputs import ModelOutput

Objective = Callable[[ModelOutput, Batch], dict[str, Tensor]]


class Trainer:
    def __init__(self, model: nn.Module, config: TrainingConfig, objective: Objective) -> None:
        self.model = model
        self.config = config
        self.objective = objective
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
        )

    def build_optimizer(self) -> AdamW:
        """Override for sparse parameters, task-specific groups or scaling rules."""
        return AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def training_step(self, batch: Batch) -> dict[str, Tensor]:
        """Return scalar losses with a differentiable 'total' entry."""
        return self.objective(self.model(batch), batch)

    def fit(
        self,
        train_loader: DataLoader,
        run_dir: Path,
        valid_loader: DataLoader | None = None,
        evaluator: Callable[..., Any] | None = None,
        evaluation_config: EvaluationConfig | None = None,
    ) -> None:
        optimizer = self.build_optimizer()
        self.model, optimizer, train_loader = self.accelerator.prepare(
            self.model, optimizer, train_loader
        )
        if valid_loader is not None:
            valid_loader = self.accelerator.prepare(valid_loader)
        optimizer.zero_grad()
        for epoch in range(self.config.epochs):
            self.model.train()
            for batch in train_loader:
                with self.accelerator.accumulate(self.model):
                    losses = self.training_step(batch)
                    if "total" not in losses or losses["total"].ndim != 0:
                        raise ValueError("The objective must return a scalar Tensor under 'total'.")
                    self.accelerator.backward(losses["total"])
                    if self.accelerator.sync_gradients and self.config.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad()
            if valid_loader is not None and evaluator is not None:
                self.model.eval()
                # Evaluators own no_grad, distributed aggregation and item-level semantics.
                metrics = evaluator(
                    self.model,
                    valid_loader,
                    accelerator=self.accelerator,
                    config=evaluation_config,
                )
                self.accelerator.print(f"epoch={epoch + 1} validation={metrics}")
            self.accelerator.save_state(str(run_dir / f"epoch_{epoch + 1:04d}"))
        self.accelerator.wait_for_everyone()
