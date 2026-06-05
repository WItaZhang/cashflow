from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from sklearn.metrics import roc_auc_score

from cashflow_ml.evaluations.metrics import group_auc, per_group_auc


def standard_holdout(test_df: pd.DataFrame, y_pred) -> dict:
    """Default competition-oriented evaluation."""
    return {
        "test_group_auc": group_auc(test_df, y_pred),
        "test_auc": float(roc_auc_score(test_df["FPF_TARGET"].astype(int), y_pred)),
        "per_client_auc": per_group_auc(test_df, y_pred),
    }


def auc_only(test_df: pd.DataFrame, y_pred) -> dict:
    return {
        "test_auc": float(roc_auc_score(test_df["FPF_TARGET"].astype(int), y_pred)),
    }


EVALUATORS: dict[str, Callable[[pd.DataFrame, object], dict]] = {
    "standard_holdout": standard_holdout,
    "auc_only": auc_only,
}


def get_evaluator(name: str) -> Callable[[pd.DataFrame, object], dict]:
    try:
        return EVALUATORS[name]
    except KeyError as exc:
        known = ", ".join(sorted(EVALUATORS))
        raise ValueError(f"Unknown evaluator '{name}'. Available: {known}") from exc
