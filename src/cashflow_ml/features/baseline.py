"""Cashflow aggregates, category spending, income gaps, and history features.

Windows include transactions exactly 7, 30, or 90 days before evaluation.
Category pivots retain the original observed-category semantics: an absent
consumer/category pair is zero within a populated pivot, while a globally
absent category is not synthesized. Missing observations remain missing.
"""

from __future__ import annotations

import pandas as pd

from cashflow_ml.features.common import safe_div

_CAT_ALL = {3, 12, 26}
_CAT_D30 = {3, 12, 23, 26}
_CAT_D90 = {3, 10, 12, 23, 26}


def _cat_pivot(
    tx: pd.DataFrame, cats: set[int], prefix: str, want_count: bool, want_amount: bool
) -> pd.DataFrame:
    """Pivot only observed categories, preserving the historical missingness."""
    sub = tx[tx["category"].isin(cats)].copy()
    if sub.empty:
        return pd.DataFrame(index=pd.Index([], name="masked_consumer_id"))
    parts = []
    if want_count:
        cnt = sub.pivot_table(
            index="masked_consumer_id",
            columns="category",
            values="masked_transaction_id",
            aggfunc="count",
            fill_value=0,
        ).rename(columns=lambda c: f"{prefix}_cat_{int(c)}_count")
        parts.append(cnt)
    if want_amount:
        amt = sub.pivot_table(
            index="masked_consumer_id",
            columns="category",
            values="amount_abs",
            aggfunc="sum",
            fill_value=0.0,
        ).rename(columns=lambda c: f"{prefix}_cat_{int(c)}_amount")
        parts.append(amt)
    result = parts[0]
    for p in parts[1:]:
        result = result.join(p, how="outer")
    return result


def _aggregate_cashflow(consumers: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the complete history and inclusive 7/30/90-day windows."""
    agg_all = tx.groupby("masked_consumer_id").agg(
        amount_mean=("amount", "mean"),
        amount_std=("amount", "std"),
        amount_abs_mean=("amount_abs", "mean"),
        amount_abs_max=("amount_abs", "max"),
        inflow_sum=("inflow_amount", "sum"),
    )
    tx7 = tx[tx["days_before"] <= 7]
    tx30 = tx[tx["days_before"] <= 30]
    tx90 = tx[tx["days_before"] <= 90]
    agg_7 = tx7.groupby("masked_consumer_id").agg(
        inflow_sum_7d=("inflow_amount", "sum"),
        txn_count_7d=("masked_transaction_id", "count"),
    )
    agg_30 = tx30.groupby("masked_consumer_id").agg(
        inflow_sum_30d=("inflow_amount", "sum"),
        txn_count_30d=("masked_transaction_id", "count"),
    )
    agg_90 = tx90.groupby("masked_consumer_id").agg(
        txn_count_90d=("masked_transaction_id", "count"),
    )

    cat_all = _cat_pivot(tx, _CAT_ALL, "all", want_count=True, want_amount=True)
    cat_d30 = _cat_pivot(tx30, _CAT_D30, "d30", want_count=False, want_amount=True)
    cat_d90 = _cat_pivot(tx90, _CAT_D90, "d90", want_count=True, want_amount=True)

    features = consumers[
        ["masked_consumer_id", "total_balance", "FPF_TARGET", "evaluation_date"]
    ].copy()
    for frame in [agg_all, agg_7, agg_30, agg_90]:
        features = features.merge(frame.reset_index(), on="masked_consumer_id", how="left")
    for pivot in [cat_all, cat_d30, cat_d90]:
        if not pivot.empty:
            features = features.merge(pivot.reset_index(), on="masked_consumer_id", how="left")

    return features


def _add_spending_ratios(features: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Normalize cashflow by balances, income, and transaction activity."""
    outflow_sum = tx.groupby("masked_consumer_id")["amount"].apply(
        lambda s: (-s.clip(upper=0)).sum()
    )
    features = features.merge(
        outflow_sum.rename("_outflow_sum").reset_index(),
        on="masked_consumer_id",
        how="left",
    )
    features["_outflow_sum"] = features["_outflow_sum"].fillna(0.0)

    bal_abs = features["total_balance"].abs() + 1.0
    features["inflow_to_balance_ratio"] = safe_div(features["inflow_sum"], bal_abs)
    features["outflow_to_balance_ratio"] = safe_div(features["_outflow_sum"], bal_abs)
    features["outflow_to_inflow_ratio"] = safe_div(features["_outflow_sum"], features["inflow_sum"])
    features["recent_7d_to_30d_txn_ratio"] = safe_div(
        features["txn_count_7d"], features["txn_count_30d"]
    )
    features["recent_30d_to_90d_txn_ratio"] = safe_div(
        features["txn_count_30d"], features["txn_count_90d"]
    )

    cc_out = (
        tx[tx["category"] == 26].groupby("masked_consumer_id")["amount_abs"].sum().rename("_cc")
    )
    features = features.merge(cc_out.reset_index(), on="masked_consumer_id", how="left")
    features["_cc"] = features["_cc"].fillna(0.0)
    features["cc_payment_amount_share"] = safe_div(features["_cc"], features["_outflow_sum"])

    gm = (
        tx[tx["category"] == 16]
        .groupby("masked_consumer_id")
        .agg(
            _gm_count=("masked_transaction_id", "count"),
            _gm_amount=("amount_abs", "sum"),
        )
    )
    features = features.merge(gm.reset_index(), on="masked_consumer_id", how="left")
    features[["_gm_count", "_gm_amount"]] = features[["_gm_count", "_gm_amount"]].fillna(0)
    features["gen_merch_avg_txn_amount"] = safe_div(features["_gm_amount"], features["_gm_count"])

    return features


def _add_income_regularity(features: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Measure positive-income gaps and the smallest positive-income month."""
    inflow_tx = tx[tx["inflow_amount"] > 0].copy()
    if not inflow_tx.empty:
        inflow_tx["_month"] = inflow_tx["posted_date"].values.astype("datetime64[M]")
        monthly_in = (
            inflow_tx.groupby(["masked_consumer_id", "_month"])["inflow_amount"]
            .sum()
            .reset_index()
            .sort_values(["masked_consumer_id", "_month"])
        )
        monthly_min = (
            monthly_in.groupby("masked_consumer_id")["inflow_amount"]
            .min()
            .rename("inflow_monthly_min")
        )
        inflow_srt = inflow_tx.sort_values(["masked_consumer_id", "posted_date"])
        inflow_srt["_gap"] = inflow_srt.groupby("masked_consumer_id")["posted_date"].diff().dt.days
        gaps = inflow_srt.groupby("masked_consumer_id")["_gap"].agg(
            inflow_gap_mean="mean", inflow_gap_std="std"
        )
        gaps["inflow_gap_cv"] = safe_div(gaps["inflow_gap_std"], gaps["inflow_gap_mean"])
        features = features.merge(
            monthly_min.reset_index(), on="masked_consumer_id", how="left"
        ).merge(
            gaps[["inflow_gap_mean", "inflow_gap_cv"]].reset_index(),
            on="masked_consumer_id",
            how="left",
        )

    return features


def _add_history_features(
    features: pd.DataFrame, consumers: pd.DataFrame, tx: pd.DataFrame
) -> pd.DataFrame:
    """Measure observation length and time since the most recent transaction."""
    conf = tx.groupby("masked_consumer_id")["posted_date"].agg(_first="min", _last="max")
    conf = conf.merge(
        consumers[["masked_consumer_id", "evaluation_date"]],
        on="masked_consumer_id",
        how="left",
    )
    conf["history_days"] = (conf["evaluation_date"] - conf["_first"]).dt.days.clip(lower=0)
    conf["recency_days"] = (conf["evaluation_date"] - conf["_last"]).dt.days.clip(lower=0)
    features = features.merge(
        conf[["masked_consumer_id", "history_days", "recency_days"]],
        on="masked_consumer_id",
        how="left",
    )

    return features


def compute_baseline_features(consumers: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Build the non-trend baseline from already prepared transactions."""
    features = _aggregate_cashflow(consumers, tx)
    features = _add_spending_ratios(features, tx)
    features = _add_income_regularity(features, tx)
    features = _add_history_features(features, consumers, tx)
    return features.drop(columns=[c for c in features.columns if c.startswith("_")])
