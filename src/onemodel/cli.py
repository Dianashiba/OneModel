"""Small shared CLI helpers; model imports are declared by the Gin recipes."""

import argparse
import json

from onemodel.config import ExperimentConfig


def make_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", help="Path to an experiment .gin file.")
    parser.add_argument("--gin", action="append", default=[], help="Gin binding; repeat as needed.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print configuration without constructing models or reading datasets.",
    )
    return parser


def print_config(experiment: ExperimentConfig) -> None:
    print(json.dumps(experiment.describe(), indent=2, ensure_ascii=False))
