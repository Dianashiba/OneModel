"""Extend the common loop with UniR2 training policies as needed."""

import gin

from onemodel.trainers.engine import Trainer


@gin.configurable
class UniR2Trainer(Trainer):
    """Placeholder recipe: implement generation warmup, joint training and parameter groups.

    Only the generic loop is inherited; the paper's two-stage schedule is not implemented.
    """
