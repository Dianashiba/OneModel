"""Extend the common loop with OneTrans-V2 training policies as needed."""

import gin

from onemodel.trainers.engine import Trainer


@gin.configurable
class OneTransV2Trainer(Trainer):
    """Placeholder recipe: add parameter groups, loss schedules and SNT handling.

    Only the generic loop is inherited; no paper-specific recipe is implemented.
    """
