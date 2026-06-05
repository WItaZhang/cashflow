from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def resolve_data_paths(config: dict) -> tuple[str, str]:
    data_cfg = config["data"]
    consumer_file = Path(data_cfg["consumer_file"]).expanduser()
    transactions_dir = Path(data_cfg["transactions_dir"]).expanduser()
    if not consumer_file.exists():
        raise FileNotFoundError(f"consumer_file not found: {consumer_file}")
    if not transactions_dir.exists():
        raise FileNotFoundError(f"transactions_dir not found: {transactions_dir}")
    return str(consumer_file), str(transactions_dir)


def per_client_stratified_622_split(
    df: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split inside each client prefix so every prefix appears in every split."""
    tr_parts, va_parts, te_parts = [], [], []
    for _, sub in df.groupby(df["masked_consumer_id"].str[:3], sort=False):
        strat = sub["FPF_TARGET"] if sub["FPF_TARGET"].nunique() > 1 else None
        train_val, test = train_test_split(
            sub,
            test_size=0.2,
            random_state=seed,
            stratify=strat,
        )
        strat2 = train_val["FPF_TARGET"] if train_val["FPF_TARGET"].nunique() > 1 else None
        train, val = train_test_split(
            train_val,
            test_size=0.25,
            random_state=seed,
            stratify=strat2,
        )
        tr_parts.append(train)
        va_parts.append(val)
        te_parts.append(test)
    return (
        pd.concat(tr_parts).reset_index(drop=True),
        pd.concat(va_parts).reset_index(drop=True),
        pd.concat(te_parts).reset_index(drop=True),
    )
