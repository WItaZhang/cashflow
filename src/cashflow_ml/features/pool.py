from __future__ import annotations

import pandas as pd

from cashflow_ml.features.feature import compute_balance_features, compute_daily_balance
from cashflow_ml.features.legacy_v16_2 import (
    _base_process_inputs,
    compute_fft_features,
    compute_risk_ts_features,
)
from cashflow_ml.features.sets import FEATURE_SETS

ID_COLUMNS = ["masked_consumer_id", "FPF_TARGET", "evaluation_date"]


def resolve_feature_names(feature_sets: list[str], extra_features: list[str] | None = None) -> list[str]:
    """Resolve readable feature-set names into a stable, de-duplicated column list."""
    names: list[str] = []
    for set_name in feature_sets:
        if set_name not in FEATURE_SETS:
            known = ", ".join(sorted(FEATURE_SETS))
            raise ValueError(f"Unknown feature set '{set_name}'. Available: {known}")
        names.extend(FEATURE_SETS[set_name])
    names.extend(extra_features or [])
    return list(dict.fromkeys(names))


def build_feature_frame(consumer_file: str, transactions_dir: str, feature_sets: list[str], extra_features: list[str] | None = None) -> pd.DataFrame:
    """Build all feature families needed by the requested feature-set names."""
    features, intermediates = _base_process_inputs(consumer_file, transactions_dir)
    requested = resolve_feature_names(feature_sets, extra_features)

    if any(name.startswith("fft_") for name in requested):
        fft = compute_fft_features(
            intermediates["inflow_monthly"],
            intermediates["outflow_monthly"],
            intermediates["net_monthly"],
        )
        features = features.merge(fft.reset_index(), on="masked_consumer_id", how="left")

    if any(name.startswith("risk_") for name in requested):
        risk = compute_risk_ts_features(
            intermediates["transactions"],
            intermediates["inflow_monthly"],
            intermediates["outflow_monthly"],
            intermediates["net_monthly"],
            intermediates["consumers"],
        )
        features = features.merge(risk.reset_index(), on="masked_consumer_id", how="left")

    if any(name.startswith("balance_") for name in requested):
        daily_bal = compute_daily_balance(
            intermediates["transactions"],
            intermediates["consumers"][["masked_consumer_id", "evaluation_date", "total_balance"]],
        )
        bal = compute_balance_features(daily_bal, intermediates["consumers"])
        features = features.merge(bal.reset_index(), on="masked_consumer_id", how="left")

    keep = [col for col in ID_COLUMNS if col in features.columns]
    keep.extend([col for col in requested if col in features.columns])
    missing = [col for col in requested if col not in features.columns]
    if missing:
        raise ValueError(f"Requested features were not produced: {missing}")
    return features[keep].copy()
