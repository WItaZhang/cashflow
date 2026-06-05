from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def group_auc(df: pd.DataFrame, y_pred) -> float:
    """Mean ROC-AUC across masked consumer prefixes."""
    tmp = df.copy()
    tmp["_group_id"] = tmp["masked_consumer_id"].str[:3]
    tmp["_y_pred"] = y_pred
    aucs = []
    for _, group in tmp.groupby("_group_id"):
        if group["FPF_TARGET"].nunique() < 2:
            continue
        aucs.append(roc_auc_score(group["FPF_TARGET"].astype(int), group["_y_pred"]))
    return float(np.mean(aucs)) if aucs else float("nan")


def per_group_auc(df: pd.DataFrame, y_pred, group_col: str = "_client") -> dict[str, float]:
    """ROC-AUC per client prefix or another prepared group column."""
    tmp = df.copy()
    tmp["_y_pred"] = y_pred
    if group_col == "_client":
        tmp[group_col] = tmp["masked_consumer_id"].str[:3]
    out = {}
    for group_id, group in tmp.groupby(group_col):
        if group["FPF_TARGET"].nunique() < 2:
            continue
        out[str(group_id)] = float(roc_auc_score(group["FPF_TARGET"].astype(int), group["_y_pred"]))
    return out
