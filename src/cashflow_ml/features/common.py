"""Shared arithmetic and point-in-time transaction preparation."""

from __future__ import annotations

import pandas as pd

# Part of the historical feature definitions; changing this changes model inputs.
EPS = 1e-6


def safe_div(a, b, eps=EPS):
    """Preserve the original additive-epsilon division and NaN propagation."""
    return a / (b + eps)


def prepare_transactions(consumers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Join evaluation dates and exclude later transactions without mutating inputs.

    The evaluation day is included. Date parsing and input validation belong to
    the data layer; all transformations here operate on in-memory DataFrames.
    """
    tx = transactions.merge(
        consumers[["masked_consumer_id", "evaluation_date"]],
        on="masked_consumer_id",
        how="inner",
    )
    tx = tx[tx["posted_date"] <= tx["evaluation_date"]].copy()
    tx["amount_abs"] = tx["amount"].abs()
    tx["inflow_amount"] = tx["amount"].clip(lower=0)
    tx["days_before"] = (tx["evaluation_date"] - tx["posted_date"]).dt.days
    return tx
