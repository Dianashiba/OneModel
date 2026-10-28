"""Implement UniR2 here; this class currently contains no network layers."""

import gin
from torch import nn

from onemodel.data.schemas import Batch
from onemodel.models.outputs import ModelOutput


@gin.configurable
class UniR2(nn.Module):
    def __init__(self, hidden_size: int = 128, num_layers: int = 2, num_heads: int = 4) -> None:
        super().__init__()
        raise NotImplementedError(
            "Implement UniR2 in models/unir2/model.py. "
            "Use --dry-run to validate the project configuration first."
        )

    def forward(self, batch: Batch) -> ModelOutput:
        raise NotImplementedError
