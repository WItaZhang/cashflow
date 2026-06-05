from __future__ import annotations

import argparse
from pathlib import Path

from cashflow_ml.experiment import run_experiment
from cashflow_ml.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a cashflow ML experiment.")
    parser.add_argument(
        "--config",
        default="configs/v15_4.yaml",
        help="Path to an experiment YAML config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    result = run_experiment(config, config_path)
    print(f"Run directory: {result['run_dir']}")


if __name__ == "__main__":
    main()
