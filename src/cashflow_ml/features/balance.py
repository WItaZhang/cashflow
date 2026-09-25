"""Stateless feature transformation functions.

Each function takes pre-joined DataFrames and returns a per-consumer Series
or DataFrame indexed by masked_consumer_id. No I/O allowed here.
"""

from __future__ import annotations

import pandas as pd


def compute_daily_balance(
    tx: pd.DataFrame,
    consumers: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct daily end-of-day balance for each consumer by back-filling
    from the evaluation-date snapshot (total_balance).

    Strategy:
        balance(t) = total_balance - sum(amounts posted after t up to eval_date)

    This anchors the series to the known snapshot so absolute values are
    meaningful (unlike a forward cumsum from 0).

    Args:
        tx: transaction rows already filtered to posted_date <= evaluation_date,
            must have columns [masked_consumer_id, posted_date, amount].
        consumers: must have columns [masked_consumer_id, evaluation_date,
            total_balance].

    Returns:
        DataFrame with columns [masked_consumer_id, date, balance], one row
        per (consumer, calendar day) from first transaction to eval_date.
    """
    snap = consumers.set_index("masked_consumer_id")["total_balance"]
    eval_dates = consumers.set_index("masked_consumer_id")["evaluation_date"]

    records: list[pd.DataFrame] = []
    for cid, grp in tx.groupby("masked_consumer_id", sort=False):
        if cid not in snap.index:
            continue
        total_bal = snap.loc[cid]
        eval_date = eval_dates.loc[cid]

        # Daily net amount (sum of all transactions on each day)
        daily = (
            grp.groupby(grp["posted_date"].dt.normalize())["amount"]
            .sum()
            .rename("daily_net")
            .sort_index()
        )

        # Full date range: first txn date → eval_date
        all_dates = pd.date_range(daily.index.min(), eval_date, freq="D")
        daily = daily.reindex(all_dates, fill_value=0.0)

        # Back-fill: balance at end of day t =
        #   total_balance - sum(daily_net for days after t up to eval_date)
        # cumsum from the right gives cumulative future amounts
        future_cumsum = daily.iloc[::-1].cumsum().iloc[::-1].shift(-1, fill_value=0.0)
        balance_series = total_bal - future_cumsum

        df = pd.DataFrame(
            {
                "masked_consumer_id": cid,
                "date": balance_series.index,
                "balance": balance_series.values,
            }
        )
        records.append(df)

    if not records:
        return pd.DataFrame(columns=["masked_consumer_id", "date", "balance"])
    return pd.concat(records, ignore_index=True)


def compute_balance_features(daily_balance: pd.DataFrame, consumers: pd.DataFrame) -> pd.DataFrame:
    """Compute per-consumer balance-based features from the daily balance series.

    Features produced:
        balance_neg_days          — number of days with balance < 0
        balance_neg_days_ratio    — balance_neg_days / total history days
        balance_min               — minimum daily balance over history
        balance_mean              — mean daily balance over history

    Args:
        daily_balance: output of compute_daily_balance.
        consumers: must have [masked_consumer_id, evaluation_date].

    Returns:
        DataFrame indexed by masked_consumer_id with the four feature columns.
    """
    grp = daily_balance.groupby("masked_consumer_id")

    neg_days = grp["balance"].apply(lambda s: (s < 0).sum()).rename("balance_neg_days")
    total_days = grp["balance"].count().rename("_total_days")
    bal_min = grp["balance"].min().rename("balance_min")
    bal_mean = grp["balance"].mean().rename("balance_mean")

    out = pd.concat([neg_days, total_days, bal_min, bal_mean], axis=1)
    out["balance_neg_days_ratio"] = out["balance_neg_days"] / out["_total_days"].clip(lower=1)
    out = out.drop(columns=["_total_days"])
    return out
