"""Minimal forward-output contract; extend state fields as the implementations need."""

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


@dataclass
class ModelOutput:
    logits: dict[str, Tensor] = field(default_factory=dict)
    auxiliary_losses: dict[str, Tensor] = field(default_factory=dict)
    states: dict[str, Any] = field(default_factory=dict)
