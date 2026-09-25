"""Behavioral risk indicators from point-in-time cashflow history.

The selected six-feature family is exposed through ``sets.py``. The additional
historical risk indicators remain available through ``extra_features``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew

from cashflow_ml.features.common import EPS, safe_div


def _velocity_features(tx: pd.DataFrame, history_days: pd.Series) -> pd.DataFrame:
    """Compare the recent 30-day rate with the consumer's lifetime rate."""
    cid_col = "masked_consumer_id"
    tx30 = tx[tx["days_before"] <= 30]
    tx_recent_in = tx30.groupby(cid_col)["inflow_amount"].sum()
    tx_recent_out = tx30.groupby(cid_col)["amount"].apply(
        lambda amounts: (-amounts.clip(upper=0)).sum()
    )
    tx_recent_cnt = tx30.groupby(cid_col)["masked_transaction_id"].count()
    tx_all_in = tx.groupby(cid_col)["inflow_amount"].sum()
    tx_all_out = tx.groupby(cid_col)["amount"].apply(lambda amounts: (-amounts.clip(upper=0)).sum())
    tx_all_cnt = tx.groupby(cid_col)["masked_transaction_id"].count()

    vel = pd.DataFrame(index=history_days.index)
    vel["_hist"] = history_days
    vel["_in_rate_life"] = safe_div(tx_all_in, vel["_hist"])
    vel["_out_rate_life"] = safe_div(tx_all_out, vel["_hist"])
    vel["_cnt_rate_life"] = safe_div(tx_all_cnt, vel["_hist"])
    vel["_in_rate_30d"] = safe_div(tx_recent_in, 30)
    vel["_out_rate_30d"] = safe_div(tx_recent_out, 30)
    vel["_cnt_rate_30d"] = safe_div(tx_recent_cnt, 30)
    vel["risk_velocity_inflow_recent"] = safe_div(vel["_in_rate_30d"], vel["_in_rate_life"])
    vel["risk_velocity_outflow_recent"] = safe_div(vel["_out_rate_30d"], vel["_out_rate_life"])
    vel["risk_velocity_txn_recent"] = safe_div(vel["_cnt_rate_30d"], vel["_cnt_rate_life"])

    return vel[
        [
            "risk_velocity_inflow_recent",
            "risk_velocity_outflow_recent",
            "risk_velocity_txn_recent",
        ]
    ]


def _monthly_statistics(
    tx: pd.DataFrame, inflow_monthly: pd.DataFrame, outflow_monthly: pd.DataFrame
) -> pd.DataFrame:
    """Summarize observed months, including activity with no positive inflow."""
    cid_col = "masked_consumer_id"
    both = (
        inflow_monthly[["masked_consumer_id", "_month", "inflow_amount"]]
        .merge(
            outflow_monthly[["masked_consumer_id", "_month", "outflow_amount"]],
            on=["masked_consumer_id", "_month"],
            how="outer",
        )
        .fillna(0.0)
    )
    both["_net_pos"] = (both["outflow_amount"] > both["inflow_amount"]).astype(int)
    overdraft_ratio = both.groupby(cid_col).agg(
        risk_overdraft_months_ratio=("_net_pos", "mean"),
    )

    # --- inflow monthly stats -------------------------------------------------
    in_m_stats = inflow_monthly.groupby(cid_col)["inflow_amount"].agg(
        _mean="mean",
        _max="max",
    )
    in_m_stats["risk_max_inflow_to_mean"] = safe_div(in_m_stats["_max"], in_m_stats["_mean"])

    def _skew(vals):
        if len(vals) < 3:
            return np.nan
        return float(scipy_skew(vals))

    in_m_skew = (
        inflow_monthly.groupby(cid_col)["inflow_amount"]
        .apply(_skew)
        .rename("risk_inflow_monthly_skew")
    )

    # --- fraction of history months with zero inflow -------------------------
    tx_ym = tx.copy()
    tx_ym["_month"] = tx_ym["posted_date"].values.astype("datetime64[M]")
    total_active_months = tx_ym.groupby(cid_col)["_month"].nunique().rename("_total_months")
    inflow_months_count = (
        inflow_monthly.groupby(cid_col)["inflow_amount"]
        .apply(lambda s: (s > 0).sum())
        .rename("_inflow_months")
    )
    months_no_in = pd.DataFrame(
        {"_total": total_active_months, "_inflow": inflow_months_count}
    ).fillna(0)
    months_no_in["risk_months_no_inflow_ratio"] = safe_div(
        months_no_in["_total"] - months_no_in["_inflow"], months_no_in["_total"]
    )

    return (
        overdraft_ratio.join(in_m_stats[["risk_max_inflow_to_mean"]], how="outer")
        .join(in_m_skew, how="outer")
        .join(months_no_in[["risk_months_no_inflow_ratio"]], how="outer")
    )


def _recent_period_ratios(tx: pd.DataFrame, consumers: pd.DataFrame) -> pd.DataFrame:
    """Retain the historical 30-day integer arithmetic used for month buckets.

    This is intentionally not calendar-month subtraction. The historical
    conversion also depends on pandas' datetime resolution; modifying it is
    an algorithm change and requires a separately evaluated experiment.
    """
    cid_col = "masked_consumer_id"
    tx_m = tx.copy()
    tx_m["_month"] = tx_m["posted_date"].values.astype("datetime64[M]")
    eval_month = consumers[[cid_col, "evaluation_date"]].copy()
    eval_month["_eval_month"] = eval_month["evaluation_date"].values.astype("datetime64[M]")
    tx_m = tx_m.merge(eval_month[[cid_col, "_eval_month"]], on=cid_col, how="left")
    # months_ago = 0 means current month
    tx_m["_months_ago"] = (
        tx_m["_eval_month"].astype("int64") - tx_m["_month"].astype("int64")
    ) // (1_000_000_000 * 60 * 60 * 24 * 30)
    tx_last3 = tx_m[tx_m["_months_ago"] < 3]
    tx_prior3 = tx_m[(tx_m["_months_ago"] >= 3) & (tx_m["_months_ago"] < 6)]

    last3_in = tx_last3.groupby(cid_col)["inflow_amount"].sum()
    prior3_in = tx_prior3.groupby(cid_col)["inflow_amount"].sum()
    last3_out = tx_last3.groupby(cid_col)["amount"].apply(
        lambda amounts: (-amounts.clip(upper=0)).sum()
    )
    prior3_out = tx_prior3.groupby(cid_col)["amount"].apply(
        lambda amounts: (-amounts.clip(upper=0)).sum()
    )

    last3m_ratio_in = safe_div(last3_in, prior3_in.reindex(last3_in.index).fillna(EPS))
    last3m_ratio_out = safe_div(last3_out, prior3_out.reindex(last3_out.index).fillna(EPS))
    last3m_ratio_in.name = "risk_last3m_inflow_vs_prior3m"
    last3m_ratio_out.name = "risk_last3m_outflow_vs_prior3m"

    last3m_out_to_in = safe_div(last3_out, last3_in.reindex(last3_out.index).fillna(EPS))
    last3m_out_to_in.name = "risk_outflow_to_inflow_last3m"

    return pd.concat([last3m_ratio_in, last3m_ratio_out, last3m_out_to_in], axis=1)


def _transaction_mix(tx: pd.DataFrame) -> pd.DataFrame:
    """Capture payment size, weekday, spending concentration, and month timing."""
    cid_col = "masked_consumer_id"
    tx_copy = tx.copy()
    tx_copy["_is_round"] = (
        (tx_copy["amount_abs"] % 50 < EPS) | (tx_copy["amount_abs"] % 100 < EPS)
    ).astype(int)
    round_ratio = tx_copy.groupby(cid_col).agg(
        _round=("_is_round", "sum"), _total=("masked_transaction_id", "count")
    )
    round_ratio["risk_round_amount_ratio"] = safe_div(round_ratio["_round"], round_ratio["_total"])

    # --- weekend transaction ratio -------------------------------------------
    tx_copy["_is_weekend"] = tx_copy["posted_date"].dt.dayofweek.isin([5, 6]).astype(int)
    weekend_ratio = tx_copy.groupby(cid_col).agg(
        _wknd=("_is_weekend", "sum"), _tot=("masked_transaction_id", "count")
    )
    weekend_ratio["risk_weekend_txn_ratio"] = safe_div(
        weekend_ratio["_wknd"], weekend_ratio["_tot"]
    )

    # --- category HHI --------------------------------------------------------
    cat_spend = tx_copy.groupby([cid_col, "category"])["amount_abs"].sum()
    cat_total = tx_copy.groupby(cid_col)["amount_abs"].sum()

    def _hhi(group):
        cid = group.index.get_level_values(cid_col)[0]
        shares = group.values / (cat_total.loc[cid] + EPS)
        return float((shares**2).sum())

    hhi = cat_spend.groupby(level=0).apply(_hhi).rename("risk_category_hhi")

    tx_copy["_day_of_month"] = tx_copy["posted_date"].dt.day
    tx_copy["_is_late"] = (tx_copy["_day_of_month"] >= 25).astype(int)
    late_ratio = tx_copy.groupby(cid_col).agg(
        _late=("_is_late", "sum"), _tota=("masked_transaction_id", "count")
    )
    late_ratio["risk_late_month_txn_ratio"] = safe_div(late_ratio["_late"], late_ratio["_tota"])

    return (
        round_ratio[["risk_round_amount_ratio"]]
        .join(weekend_ratio[["risk_weekend_txn_ratio"]], how="outer")
        .join(hhi, how="outer")
        .join(late_ratio[["risk_late_month_txn_ratio"]], how="outer")
    )


def _normalized_income_gap(tx: pd.DataFrame, history_days: pd.Series) -> pd.Series:
    """Scale the largest interval between positive inflows by history length."""
    cid_col = "masked_consumer_id"
    inflow_tx = tx[tx["inflow_amount"] > 0].copy()
    inflow_sorted = inflow_tx.sort_values([cid_col, "posted_date"])
    inflow_sorted["_gap"] = inflow_sorted.groupby(cid_col)["posted_date"].diff().dt.days
    max_gap = inflow_sorted.groupby(cid_col)["_gap"].max().rename("_max_gap")
    hist_d = history_days
    gap_norm = safe_div(max_gap, hist_d.reindex(max_gap.index).clip(lower=1))
    gap_norm.name = "risk_inflow_gap_max_norm"

    return gap_norm


def compute_risk_ts_features(
    tx: pd.DataFrame,
    inflow_monthly: pd.DataFrame,
    outflow_monthly: pd.DataFrame,
    net_monthly: pd.DataFrame,
    consumers: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble behavioral indicators from prepared, cutoff-filtered inputs.

    ``net_monthly`` is retained in the family API so all monthly families accept
    the same series tuple; the risk formulas use inflow and outflow directly.
    """
    cid_col = "masked_consumer_id"
    first = tx.groupby(cid_col)["posted_date"].min().rename("_first")
    history = first.to_frame().merge(
        consumers[[cid_col, "evaluation_date"]], on=cid_col, how="left"
    )
    history_days = (history["evaluation_date"] - history["_first"]).dt.days.clip(lower=1)
    history_days.index = history[cid_col]
    return (
        _velocity_features(tx, history_days)
        .join(_monthly_statistics(tx, inflow_monthly, outflow_monthly), how="outer")
        .join(_recent_period_ratios(tx, consumers), how="outer")
        .join(_transaction_mix(tx), how="outer")
        .join(_normalized_income_gap(tx, history_days), how="outer")
        .drop(columns=[cid_col], errors="ignore")
    )
