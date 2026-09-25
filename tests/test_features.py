"""Feature contracts: time boundaries, historical missingness, and stable inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cashflow_ml.features import build_feature_frame, resolve_feature_names
from cashflow_ml.features.balance import compute_balance_features, compute_daily_balance
from cashflow_ml.features.common import prepare_transactions
from cashflow_ml.features.temporal import build_monthly_series, compute_fft_features


@pytest.fixture
def inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    consumers = pd.DataFrame(
        {
            "masked_consumer_id": ["a", "b", "c"],
            "evaluation_date": pd.to_datetime(["2024-06-30"] * 3),
            "FPF_TARGET": [0.0, 1.0, np.nan],
            "total_balance": [200.0, -50.0, 0.0],
        }
    )
    rows = []
    offsets = [0, 1, 6, 7, 8, 29, 30, 31, 60, 90, 91, 120, 150, 180]
    categories = [3, 12, 23, 26, 10, 16]
    for client in ("a", "b"):
        for index, days in enumerate(offsets):
            rows.append(
                {
                    "masked_consumer_id": client,
                    "masked_transaction_id": f"{client}-{index}",
                    "posted_date": pd.Timestamp("2024-06-30") - pd.Timedelta(days=days),
                    "amount": float(100 + index * 10) * (-1 if index % 3 == 0 else 1),
                    "category": categories[index % len(categories)],
                }
            )
    return consumers, pd.DataFrame(rows)


def test_feature_selection_is_ordered_and_deduplicated():
    baseline = resolve_feature_names(["cashflow_44"])
    selected = resolve_feature_names(["cashflow_56", "cashflow_44"], ["balance_mean"])
    assert len(baseline) == 44
    assert len(selected) == 57
    assert selected[:44] == baseline
    assert selected[-1] == "balance_mean"
    assert selected[:-1] == resolve_feature_names(["v15_4_keep_56"])
    assert resolve_feature_names(["cashflow_60"]) == resolve_feature_names(["v15_4_keep_60"])
    with pytest.raises(ValueError, match="Unknown feature set"):
        resolve_feature_names(["typo"])


def test_cutoff_excludes_future_and_unknown_consumers_without_mutation(inputs):
    consumers, transactions = inputs
    consumers_before, transactions_before = (
        consumers.copy(deep=True),
        transactions.copy(deep=True),
    )
    expected = build_feature_frame(consumers, transactions, ["cashflow_60"])
    future = transactions.iloc[[0]].copy()
    future["posted_date"] = pd.Timestamp("2024-07-01")
    future["amount"] = 1e12
    unknown = transactions.iloc[[0]].copy()
    unknown["masked_consumer_id"] = "not-in-consumers"
    contaminated = pd.concat([transactions, future, unknown], ignore_index=True)
    actual = build_feature_frame(consumers, contaminated, ["cashflow_60"])
    assert_frame_equal(actual, expected, check_exact=True)
    assert_frame_equal(consumers, consumers_before, check_exact=True)
    assert_frame_equal(transactions, transactions_before, check_exact=True)
    selected = resolve_feature_names(["cashflow_60"])
    assert actual[selected].columns.tolist() == selected
    assert actual["masked_consumer_id"].tolist() == ["a", "b", "c"]
    # Empty prior-month windows must not introduce metadata into risk features.
    assert actual["evaluation_date"].tolist() == consumers["evaluation_date"].tolist()


def test_recent_windows_include_their_boundary_day(inputs):
    consumers, transactions = inputs
    frame = build_feature_frame(
        consumers,
        transactions,
        [],
        [
            "inflow_sum_7d",
            "inflow_sum_30d",
            "txn_count_7d",
            "txn_count_30d",
            "txn_count_90d",
        ],
    ).set_index("masked_consumer_id")
    assert frame.loc["a", "txn_count_7d"] == 4
    assert frame.loc["a", "txn_count_30d"] == 7
    assert frame.loc["a", "txn_count_90d"] == 10
    assert frame.loc["a", "inflow_sum_7d"] == 230.0
    assert frame.loc["a", "inflow_sum_30d"] == 520.0
    # A consumer with no history remains missing, ready for train-only imputation.
    assert (
        frame.loc["c"].isna().drop(labels=["FPF_TARGET", "evaluation_date"], errors="ignore").all()
    )


def test_metadata_survives_an_empty_prior_month_window(inputs):
    consumers, transactions = inputs
    recent = transactions.loc[transactions["posted_date"] >= pd.Timestamp("2024-06-01")]
    frame = build_feature_frame(consumers, recent, ["cashflow_56"])
    assert frame["evaluation_date"].tolist() == consumers["evaluation_date"].tolist()
    assert not any(column.endswith(("_x", "_y")) for column in frame.columns)


def test_sparse_category_pivots_keep_zero_and_missing_distinct(inputs):
    consumers, transactions = inputs
    transactions = transactions.loc[
        ~((transactions["masked_consumer_id"] == "b") & (transactions["category"] == 26))
    ]
    frame = build_feature_frame(consumers, transactions, [], ["all_cat_26_amount"])
    values = frame.set_index("masked_consumer_id")["all_cat_26_amount"]
    assert values.loc["a"] > 0
    assert values.loc["b"] == 0
    assert pd.isna(values.loc["c"])


def test_globally_absent_requested_categories_are_reported(inputs):
    consumers, transactions = inputs
    transactions = transactions.loc[transactions["category"] != 26]
    with pytest.raises(ValueError, match="Requested features were not produced.*all_cat_26_amount"):
        build_feature_frame(consumers, transactions, [], ["all_cat_26_amount"])


def test_monthly_grid_keeps_observed_months_only(inputs):
    consumers, transactions = inputs
    transactions = transactions.iloc[:2].copy()
    transactions["posted_date"] = pd.to_datetime(["2024-01-15", "2024-03-15"])
    prepared = prepare_transactions(consumers, transactions)
    monthly = build_monthly_series(prepared)
    assert monthly[0]["_month"].dt.month.tolist() == [1, 3]
    assert len(monthly[0]) == 2
    # FFT requires three observed months; a gap is not a zero-filled third month.
    assert compute_fft_features(*monthly).isna().all().all()


def test_balance_is_anchored_to_evaluation_snapshot():
    consumers = pd.DataFrame(
        {
            "masked_consumer_id": ["a"],
            "evaluation_date": pd.to_datetime(["2024-01-03"]),
            "total_balance": [-10.0],
        }
    )
    transactions = pd.DataFrame(
        {
            "masked_consumer_id": ["a", "a"],
            "posted_date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "amount": [100.0, -50.0],
        }
    )
    daily = compute_daily_balance(transactions, consumers)
    assert daily["balance"].tolist() == [40.0, 40.0, -10.0]
    values = compute_balance_features(daily, consumers).loc["a"]
    assert values["balance_neg_days"] == 1
    assert values["balance_neg_days_ratio"] == pytest.approx(1 / 3)
    assert values["balance_min"] == -10.0
    assert values["balance_mean"] == pytest.approx(70 / 3)
