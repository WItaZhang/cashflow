"""Generate reproducible synthetic inputs and exercise the normal experiment path.

Labels are independent of transaction behavior. Demo metrics verify execution;
they are not estimates of credit-risk performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from cashflow_ml.config import load_config
from cashflow_ml.utils import set_thread_env

if TYPE_CHECKING:
    import pandas as pd


def build_demo_frames(config: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create an in-memory fixture using only generator settings from YAML."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    evaluation_date = pd.Timestamp(config["evaluation_date"])
    categories = config["categories"]
    periods = int(config["history_periods"])
    days_per_period = int(config["days_per_period"])
    transactions_per_period = int(config["transactions_per_period"])
    if periods < 1 or days_per_period < 1:
        raise ValueError("Demo history_periods and days_per_period must be positive.")
    if not categories or transactions_per_period < len(categories):
        raise ValueError("Demo needs at least one transaction per category per period.")

    consumers = []
    transactions = []
    for client, settings in config["clients"].items():
        count = int(settings["n_consumers"])
        fraction = float(settings["positive_fraction"])
        if count < 1 or not 0 < fraction < 1:
            raise ValueError("Demo clients need positive counts and a label fraction in (0, 1).")
        n_positive = round(count * fraction)
        labels = np.zeros(count, dtype=int)
        labels[:n_positive] = 1
        rng.shuffle(labels)

        for index, target in enumerate(labels):
            consumer_id = f"{client}_demo_{index:05d}"
            consumers.append(
                {
                    "masked_consumer_id": consumer_id,
                    "evaluation_date": evaluation_date,
                    "total_balance": rng.uniform(*settings["balance_range"]),
                    "FPF_TARGET": int(target),
                }
            )
            amount_scale = float(settings["amount_scale"]) * rng.uniform(
                *config["consumer_amount_scale_range"]
            )
            for period in range(periods):
                # Include every configured category in each period, then vary
                # the remaining transactions without relying on chance coverage.
                choices = list(range(len(categories)))
                choices.extend(
                    rng.integers(0, len(categories), size=transactions_per_period - len(categories))
                )
                rng.shuffle(choices)
                for ordinal, category_index in enumerate(choices):
                    category = categories[category_index]
                    age_days = period * days_per_period + int(rng.integers(days_per_period))
                    transactions.append(
                        {
                            "masked_consumer_id": consumer_id,
                            "masked_transaction_id": f"{consumer_id}_{period:03d}_{ordinal:03d}",
                            "posted_date": evaluation_date - pd.Timedelta(days=age_days),
                            "amount": round(
                                rng.uniform(*category["amount_range"]) * amount_scale, 2
                            ),
                            "category": int(category["id"]),
                        }
                    )

    return pd.DataFrame(consumers), pd.DataFrame(transactions)


def write_demo_inputs(config: dict) -> tuple[int, int]:
    """Write only generated staging inputs; reuse identical existing fixtures."""
    import pandas as pd

    demo_config = config["demo"]
    staging_root = Path(demo_config["staging_root"]).expanduser().resolve()
    consumer_file = Path(config["data"]["consumer_file"]).expanduser().resolve()
    transactions_dir = Path(config["data"]["transactions_dir"]).expanduser().resolve()
    for path in (consumer_file, transactions_dir):
        if not path.is_relative_to(staging_root) or path == staging_root:
            raise ValueError(f"Demo output must be below demo.staging_root: {path}")

    consumers, transactions = build_demo_frames(demo_config, int(config["seed"]))
    transaction_file = transactions_dir / "transactions_demo.parquet"
    if consumer_file == transaction_file:
        raise ValueError("Demo consumer and transaction outputs must be different files.")
    outputs = ((consumer_file, consumers), (transaction_file, transactions))
    other_shards = set(transactions_dir.glob("transactions_*.parquet")) - {transaction_file}
    if other_shards:
        raise FileExistsError(
            "Demo transaction directory contains other shards; choose a fresh staging directory."
        )
    # Check all existing files before writing anything, including on a rerun.
    for path, frame in outputs:
        if path.exists() and not pd.read_parquet(path).equals(frame):
            raise FileExistsError(
                f"Refusing to replace different data at {path}. Choose a fresh staging path."
            )
    for path, frame in outputs:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
    return len(consumers), len(transactions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo.yaml", help="Demo experiment YAML.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    set_thread_env(int(config["runtime"]["n_jobs"]))

    from cashflow_ml.experiment import run_experiment

    n_consumers, n_transactions = write_demo_inputs(config)
    print(f"Synthetic demo: {n_consumers} consumers; {n_transactions} transactions.")
    print("Labels are synthetic. Resulting metrics are not credit-risk performance evidence.")
    result = run_experiment(config, config_path)
    print(f"Run directory: {result['run_dir']}")


if __name__ == "__main__":
    main()
