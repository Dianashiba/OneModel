"""UniR2 batch assembly extension point."""

import gin

from onemodel.data.schemas import Batch, Exposure


@gin.configurable
class UniR2Collator:
    def __call__(self, samples: list[Exposure]) -> Batch:
        raise NotImplementedError(
            "Assemble user context, SID targets, item features and task-specific loss masks."
        )
