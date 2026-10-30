"""Exercise Gin recipe wiring and CLI behavior without implementing recommendation models."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import gin

from onemodel.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class ConfigAndCLITest(unittest.TestCase):
    def tearDown(self) -> None:
        gin.clear_config()

    def run_cli(self, module: str, recipe: str, *extra: str) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["ACCELERATE_USE_CPU"] = "true"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    module,
                    str(ROOT / "configs" / recipe / "base.gin"),
                    *extra,
                ],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertFalse((Path(directory) / "outputs").exists())
            return result

    def test_both_recipes_parse_and_reset_bindings(self) -> None:
        first = load_config(
            ROOT / "configs/onetrans_v2/base.gin", ["TrainingConfig.seed = 43"]
        )
        self.assertEqual(first.training.seed, 43)
        self.assertEqual(first.dataset.__name__, "UserWindowDataset")
        second = load_config(ROOT / "configs/unir2/base.gin")
        self.assertEqual(second.training.seed, 42)
        self.assertEqual(second.model.__name__, "UniR2")
        self.assertEqual(second.dataset.__name__, "ExposureDataset")

    def test_dry_run_from_another_directory_without_building_stubs(self) -> None:
        for recipe in ("onetrans_v2", "unir2"):
            with self.subTest(recipe=recipe):
                result = self.run_cli(
                    "onemodel.train", recipe, "--dry-run", "--gin", "DataConfig.batch_size = 7"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("ERROR", result.stderr)
                summary = json.loads(result.stdout)
                self.assertEqual(summary["status"], "configuration_valid")
                self.assertEqual(summary["experiment"], recipe)
                self.assertEqual(summary["data"]["batch_size"], 7)

    def test_evaluation_dry_run_accepts_checkpoint_without_reading_it(self) -> None:
        result = self.run_cli(
            "onemodel.evaluate", "unir2", "--dry-run", "--checkpoint", "example-checkpoint"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout)["evaluation"]["checkpoint"], "example-checkpoint"
        )

    def test_unknown_model_parameter_is_rejected(self) -> None:
        original_directory = Path.cwd()
        with self.assertRaises(ValueError):
            load_config(
                ROOT / "configs/unir2/base.gin", ["UniR2.unknown_parameter = 128"]
            )
        self.assertEqual(Path.cwd(), original_directory)

    def test_real_training_reports_the_unimplemented_model(self) -> None:
        result = self.run_cli("onemodel.train", "unir2")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Implement UniR2", result.stderr)

    def test_real_evaluation_requires_checkpoint(self) -> None:
        result = self.run_cli("onemodel.evaluate", "unir2")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--checkpoint", result.stderr)


if __name__ == "__main__":
    unittest.main()
