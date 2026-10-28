"""OneTrans-V2 batch assembly extension point."""

import gin

from onemodel.data.schemas import Batch, UserWindow


@gin.configurable
class OneTransV2Collator:
    def __call__(self, samples: list[UserWindow]) -> Batch:
        raise NotImplementedError(
            "Assemble shared histories, stage features, request anchors and loss masks."
        )
