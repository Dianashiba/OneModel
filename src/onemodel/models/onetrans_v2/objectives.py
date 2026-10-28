"""OneTrans-V2 objective extension point; no paper losses are implemented."""

import gin
from torch import Tensor

from onemodel.data.schemas import Batch
from onemodel.models.outputs import ModelOutput


@gin.configurable
def onetrans_v2_objective(output: ModelOutput, batch: Batch) -> dict[str, Tensor]:
    raise NotImplementedError(
        "Return scalar 'total' plus retrieval, pre-rank, fine-rank and distillation losses."
    )
