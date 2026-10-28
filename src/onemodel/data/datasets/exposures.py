"""Dataset extension points. No dataset format or sampling policy is implemented yet."""

from pathlib import Path

import gin
from torch.utils.data import Dataset

from onemodel.data.schemas import Exposure, UserWindow


@gin.configurable
class ExposureDataset(Dataset[Exposure]):
    def __init__(self, root: str, split: str) -> None:
        self.root = Path(root)
        self.split = split
        raise NotImplementedError("Implement ExposureDataset for your exposure logs and features.")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Exposure:
        raise NotImplementedError


@gin.configurable
class UserWindowDataset(Dataset[UserWindow]):
    def __init__(self, root: str, split: str) -> None:
        self.root = Path(root)
        self.split = split
        raise NotImplementedError("Implement UserWindowDataset with per-exposure history anchors.")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> UserWindow:
        raise NotImplementedError
