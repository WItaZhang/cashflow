"""Command-line entry point; experiment choices come from YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

from cashflow_ml.config import load_config
from cashflow_ml.utils import set_thread_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a cashflow credit-risk experiment.")
    parser.add_argument(
        "--config", default="configs/cashflow_moe.yaml", help="Experiment YAML path."
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    # Set native-library thread limits before importing the numeric stack.
    set_thread_env(config.get("runtime", {}).get("n_jobs", 8))
    from cashflow_ml.experiment import run_experiment

    result = run_experiment(config, config_path)
    print(f"Run directory: {result['run_dir']}")


if __name__ == "__main__":
    main()
