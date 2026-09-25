"""Read and validate input tables; split consumers without fitting transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CONSUMER_COLUMNS = ("masked_consumer_id", "evaluation_date", "total_balance", "FPF_TARGET")
TRANSACTION_COLUMNS = (
    "masked_consumer_id",
    "masked_transaction_id",
    "posted_date",
    "amount",
    "category",
)


def resolve_data_paths(config: dict) -> tuple[Path, Path]:
    """Paths remain relative to the working directory, as in the original CLI."""
    consumer_file = Path(config["data"]["consumer_file"]).expanduser()
    transactions_dir = Path(config["data"]["transactions_dir"]).expanduser()
    if not consumer_file.is_file():
        raise FileNotFoundError(f"consumer_file not found: {consumer_file}")
    if not transactions_dir.is_dir():
        raise FileNotFoundError(f"transactions_dir not found: {transactions_dir}")
    return consumer_file, transactions_dir


def transaction_files(transactions_dir: Path) -> list[Path]:
    files = sorted(transactions_dir.glob("transactions_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No transactions_*.parquet files in {transactions_dir}")
    return files


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_inputs(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load external snapshots and ordered shards without modifying input files."""
    consumer_file, transactions_dir = resolve_data_paths(config)
    files = transaction_files(transactions_dir)
    consumers = pd.read_parquet(consumer_file)
    transactions = pd.concat(
        [pd.read_parquet(path) for path in files],
        ignore_index=True,
    )
    _require_columns(consumers, CONSUMER_COLUMNS, "Consumer table")
    _require_columns(transactions, TRANSACTION_COLUMNS, "Transaction table")
    if consumers.empty or transactions.empty:
        raise ValueError("Consumer and transaction tables must both contain rows.")
    for name, frame in (("Consumer", consumers), ("Transaction", transactions)):
        ids = frame["masked_consumer_id"]
        if (
            ids.isna().any()
            or not ids.map(lambda value: isinstance(value, str) and len(value) >= 3).all()
        ):
            raise ValueError(f"{name} IDs require a three-character client prefix.")
    if consumers["masked_consumer_id"].duplicated().any():
        raise ValueError("Consumer table must have exactly one row per masked_consumer_id.")
    if not consumers["FPF_TARGET"].dropna().isin([0, 1]).all():
        raise ValueError("FPF_TARGET must be 0, 1, or null for unlabeled consumers.")
    for frame, column in ((consumers, "evaluation_date"), (transactions, "posted_date")):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
        if frame[column].isna().any():
            raise ValueError(f"{column} cannot contain null dates.")
    for frame, column in ((consumers, "total_balance"), (transactions, "amount")):
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"{column} must be numeric.")
        if np.isinf(frame[column].to_numpy(dtype=float)).any():
            raise ValueError(f"{column} cannot contain infinite values.")
    return consumers, transactions


def per_client_stratified_622_split(
    df: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preserve the original per-client 60/20/20 split and RNG sequence."""
    if df.empty:
        raise ValueError("No labeled consumers are available for splitting.")
    if df["masked_consumer_id"].duplicated().any():
        raise ValueError("Split input must contain one row per consumer.")
    train_parts, validation_parts, test_parts = [], [], []
    for client, subset in df.groupby(df["masked_consumer_id"].str[:3], sort=False):
        stratify = subset["FPF_TARGET"] if subset["FPF_TARGET"].nunique() > 1 else None
        try:
            train_val, test = train_test_split(
                subset,
                test_size=0.2,
                random_state=seed,
                stratify=stratify,
            )
            stratify = train_val["FPF_TARGET"] if train_val["FPF_TARGET"].nunique() > 1 else None
            train, validation = train_test_split(
                train_val,
                test_size=0.25,
                random_state=seed,
                stratify=stratify,
            )
        except ValueError as error:
            raise ValueError(
                f"Cannot create a stratified 60/20/20 split for client {client}: {error}"
            ) from error
        train_parts.append(train)
        validation_parts.append(validation)
        test_parts.append(test)
    return tuple(
        pd.concat(parts).reset_index(drop=True)
        for parts in (train_parts, validation_parts, test_parts)
    )


def split_manifest(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    """Record membership so each reported score can be traced to its consumers."""
    parts = []
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        part = frame[["masked_consumer_id"]].copy()
        part["client"] = part["masked_consumer_id"].str[:3]
        part["split"] = name
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
