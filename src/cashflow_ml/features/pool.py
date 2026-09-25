from __future__ import annotations

import pandas as pd

from cashflow_ml.features.balance import compute_balance_features, compute_daily_balance
from cashflow_ml.features.baseline import compute_baseline_features
from cashflow_ml.features.common import prepare_transactions
from cashflow_ml.features.risk import compute_risk_ts_features
from cashflow_ml.features.sets import FEATURE_SETS
from cashflow_ml.features.temporal import (
    build_monthly_series,
    compute_fft_features,
    compute_trend_features,
)

ID_COLUMNS = ["masked_consumer_id", "FPF_TARGET", "evaluation_date"]


def resolve_feature_names(
    feature_sets: list[str], extra_features: list[str] | None = None
) -> list[str]:
    """Resolve readable feature-set names into a stable, de-duplicated column list."""
    names: list[str] = []
    for set_name in feature_sets:
        if set_name not in FEATURE_SETS:
            known = ", ".join(sorted(FEATURE_SETS))
            raise ValueError(f"Unknown feature set '{set_name}'. Available: {known}")
        names.extend(FEATURE_SETS[set_name])
    names.extend(extra_features or [])
    return list(dict.fromkeys(names))


def build_feature_frame(
    consumers: pd.DataFrame,
    transactions: pd.DataFrame,
    feature_sets: list[str],
    extra_features: list[str] | None = None,
) -> pd.DataFrame:
    """Transform parsed input frames into an ordered, point-in-time feature table.

    Inputs are never mutated. Feature functions perform no I/O, fitting, or
    imputation; the training layer owns any learned preprocessing.
    """
    requested = resolve_feature_names(feature_sets, extra_features)
    tx = prepare_transactions(consumers, transactions)
    features = compute_baseline_features(consumers, tx)
    monthly = build_monthly_series(tx)
    trends = compute_trend_features(*monthly)
    features = features.merge(trends.reset_index(), on="masked_consumer_id", how="left")

    if any(name.startswith("fft_") for name in requested):
        fft = compute_fft_features(*monthly)
        features = features.merge(fft.reset_index(), on="masked_consumer_id", how="left")

    if any(name.startswith("risk_") for name in requested):
        risk = compute_risk_ts_features(tx, *monthly, consumers)
        features = features.merge(risk.reset_index(), on="masked_consumer_id", how="left")

    if any(name.startswith("balance_") for name in requested):
        daily_bal = compute_daily_balance(
            tx,
            consumers[["masked_consumer_id", "evaluation_date", "total_balance"]],
        )
        bal = compute_balance_features(daily_bal, consumers)
        features = features.merge(bal.reset_index(), on="masked_consumer_id", how="left")

    keep = [col for col in ID_COLUMNS if col in features.columns]
    keep.extend([col for col in requested if col in features.columns])
    missing = [col for col in requested if col not in features.columns]
    if missing:
        raise ValueError(f"Requested features were not produced: {missing}")
    return features[keep].copy()
