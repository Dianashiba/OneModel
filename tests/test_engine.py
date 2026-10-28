"""Test shared training infrastructure with a tiny scalar fixture, not a recommender."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("ACCELERATE_USE_CPU", "true")

import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from onemodel.config import TrainingConfig
from onemodel.models.outputs import ModelOutput
from onemodel.trainers.engine import Trainer


class ScalarFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch: dict) -> ModelOutput:
        return ModelOutput(logits={"score": self.weight * batch["feature"]})


def fixture_objective(output: ModelOutput, batch: dict) -> dict:
    return {"total": (output.logits["score"] - batch["label"]).square().mean()}


class EngineTest(unittest.TestCase):
    def test_accumulation_updates_weights_and_checkpoint_loads_for_evaluation(self) -> None:
        model = ScalarFixture()
        trainer = Trainer(
            model,
            TrainingConfig(epochs=1, learning_rate=0.1, gradient_accumulation_steps=2),
            fixture_objective,
        )
        dataloader = DataLoader(
            [{"feature": 1.0, "label": 1.0}, {"feature": 2.0, "label": 2.0}],
            batch_size=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer.fit(dataloader, Path(directory))
            expected_weight = trainer.accelerator.unwrap_model(trainer.model).weight.detach().clone()
            self.assertGreater(expected_weight.item(), 0)
            checkpoint = Path(directory) / "epoch_0001"
            self.assertTrue(checkpoint.is_dir())
            evaluator = Accelerator()
            restored = evaluator.prepare(ScalarFixture())
            evaluator.load_state(str(checkpoint))
            torch.testing.assert_close(restored.weight, expected_weight)


if __name__ == "__main__":
    unittest.main()
