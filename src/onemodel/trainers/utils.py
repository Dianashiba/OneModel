"""Experiment provenance and shared run directories."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import gin
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

from onemodel.config import ExperimentConfig


def create_run_dir(experiment: ExperimentConfig, accelerator: Accelerator) -> Path:
    """Choose the run ID on rank zero so every worker uses the same output path."""
    paths = [None]
    if accelerator.is_main_process:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = Path(experiment.output_dir).resolve() / experiment.name / run_id
        path.mkdir(parents=True, exist_ok=False)
        paths[0] = str(path)
    broadcast_object_list(paths)
    accelerator.wait_for_everyone()
    return Path(paths[0])


def save_run_metadata(
    run_dir: Path, experiment: ExperimentConfig, config_path: str, bindings: list[str]
) -> None:
    """Call on the main process after constructing components to capture Gin defaults."""
    repository = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=False
    )
    metadata = experiment.describe()
    metadata.pop("status")
    metadata.update(
        config_path=str(Path(config_path).resolve()),
        bindings=bindings,
        git_sha=result.stdout.strip() if result.returncode == 0 else None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    (run_dir / "config.gin").write_text(gin.config_str() + "\n", encoding="utf-8")
    (run_dir / "operative.gin").write_text(gin.operative_config_str() + "\n", encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
