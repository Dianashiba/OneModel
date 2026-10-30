"""Gin configuration shared by training and evaluation entry points."""

import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gin


@gin.configurable
@dataclass
class DataConfig:
    name: str = "example"
    root: str = "datasets/processed"
    train_split: str = "train"
    valid_split: str = "valid"
    test_split: str = "test"
    batch_size: int = 32
    num_workers: int = 0


@gin.configurable
@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: str = "no"


@gin.configurable
@dataclass
class EvaluationConfig:
    top_k: tuple[int, ...] = (5, 10)
    split: str = "test"
    checkpoint: str | None = None


@gin.configurable
@dataclass
class ExperimentConfig:
    name: str = gin.REQUIRED
    model: Callable[..., Any] = gin.REQUIRED
    dataset: Callable[..., Any] = gin.REQUIRED
    collator: Callable[..., Any] = gin.REQUIRED
    objective: Callable[..., Any] = gin.REQUIRED
    trainer: Callable[..., Any] = gin.REQUIRED
    evaluator: Callable[..., Any] = gin.REQUIRED
    data: DataConfig = gin.REQUIRED
    training: TrainingConfig = gin.REQUIRED
    evaluation: EvaluationConfig = gin.REQUIRED
    output_dir: str = "outputs"

    def validate(self) -> None:
        if not self.name or Path(self.name).name != self.name or self.name in {".", ".."}:
            raise ValueError("ExperimentConfig.name must be a non-empty directory name.")
        for name in ("model", "dataset", "collator", "objective", "trainer", "evaluator"):
            if not callable(getattr(self, name)):
                raise ValueError(f"ExperimentConfig.{name} must be a Gin callable reference.")
        if self.data.batch_size < 1 or self.data.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers must be non-negative.")
        if self.training.epochs < 1 or self.training.learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive.")
        if self.training.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.training.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be 'no', 'fp16', or 'bf16'.")
        if not self.evaluation.top_k or any(k < 1 for k in self.evaluation.top_k):
            raise ValueError("evaluation.top_k must contain positive integers.")

    def describe(self) -> dict[str, Any]:
        return {
            "status": "configuration_valid",
            "experiment": self.name,
            "components": {
                name: f"{factory.__module__}.{factory.__qualname__}"
                for name in ("model", "dataset", "collator", "objective", "trainer", "evaluator")
                for factory in (getattr(self, name),)
            },
            "data": asdict(self.data),
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation),
            "output_dir": self.output_dir,
        }


def load_config(path: str | Path, bindings: Sequence[str] = ()) -> ExperimentConfig:
    """Resolve includes from the config directory and its parent; reject unknown bindings."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Gin config does not exist: {config_path}")
    gin.clear_config()
    gin.add_config_file_search_path(str(config_path.parent))
    gin.add_config_file_search_path(str(config_path.parent.parent))
    # Gin resolves includes against the process directory. Limit the directory
    # change to parsing and restore it before constructing any components.
    original_directory = Path.cwd()
    try:
        os.chdir(config_path.parent)
        gin.parse_config_files_and_bindings([str(config_path)], list(bindings), skip_unknown=False)
    finally:
        os.chdir(original_directory)
    experiment = ExperimentConfig()
    experiment.validate()
    return experiment
