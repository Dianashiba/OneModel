"""Unified retrieval-to-ranking evaluator extension point."""

import gin
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from onemodel.config import EvaluationConfig


@gin.configurable
def evaluate_pipeline(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    accelerator: Accelerator,
    config: EvaluationConfig,
) -> dict[str, float]:
    raise NotImplementedError(
        "Implement the evaluation protocol, candidate generation and distributed metric aggregation."
    )
