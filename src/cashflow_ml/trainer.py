"""Fit, evaluate, and persist the cashflow mixture of experts."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cashflow_ml.evaluations import get_evaluator
from cashflow_ml.models.lgbm_factory import build_lgbm
from cashflow_ml.models.moe_lgbm import SoftRoutingLgbmMoe
from cashflow_ml.sampling import resample_group, sample_weights

log = logging.getLogger(__name__)


def fit_model(
    model: SoftRoutingLgbmMoe,
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> SoftRoutingLgbmMoe:
    """Fit the router on all training rows, then fit each eligible expert.

    Splitting must happen before this call. Resampling touches each expert's
    training subset only; the router sees the original group distribution.
    Expert ordering and the shared random generator preserve v15_4 behavior.
    """
    _validate_training_frame(train_df, feature_cols)
    model.feature_cols = list(feature_cols)
    model.router = None
    model.router_classes = None
    model.experts.clear()
    model.expert_cols.clear()
    X_train = train_df[model.feature_cols]
    y_train = train_df["FPF_TARGET"].astype(int)
    clients = train_df["masked_consumer_id"].str[:3].values
    hp = model.config["hyperparameters"]
    model.router = build_lgbm(hp["router"], model.seed, model.n_jobs)
    model.router.fit(X_train, clients)
    model.router_classes = np.array(model.router.named_steps["clf"].classes_)

    training = model.config.get("training", {})
    rng = np.random.default_rng(model.seed)
    for client, expert_hp in hp["experts"].items():
        mask = clients == client
        sub_y = y_train.iloc[mask]
        if sub_y.nunique() < 2 or mask.sum() < training.get("min_expert_rows", 20):
            log.warning(
                "Skipping expert %s: rows=%d, classes=%d", client, mask.sum(), sub_y.nunique()
            )
            continue
        X_sub, y_sub = resample_group(
            X_train.iloc[mask].copy(),
            sub_y.copy(),
            client,
            training,
            rng,
        )
        if y_sub.nunique() < 2:
            raise ValueError(
                f"Resampling removed a class from expert {client!r}; "
                "increase downsample_negative_keep_ratio or provide more training rows."
            )
        weights = sample_weights(y_sub, training)
        expert = build_lgbm(expert_hp, model.seed, model.n_jobs)
        fit_kwargs = {} if weights is None else {"clf__sample_weight": weights}
        expert.fit(X_sub, y_sub, **fit_kwargs)
        model.experts[client] = expert
        model.expert_cols[client] = list(model.feature_cols)
        log.info(
            "Fitted expert %s: original_rows=%d, resampled_rows=%d", client, mask.sum(), len(y_sub)
        )
    if not model.experts:
        raise ValueError(
            "No eligible experts: check client prefixes, class counts, and min_expert_rows."
        )
    return model


def evaluate_model(
    model: SoftRoutingLgbmMoe,
    frame: pd.DataFrame,
    evaluator_name: str = "standard_holdout",
) -> dict:
    """Evaluate without fitting or changing any model state."""
    return get_evaluator(evaluator_name)(frame, model.predict_proba(frame))


def save_model(model: SoftRoutingLgbmMoe, path: str | Path) -> None:
    """Persist the fitted router, experts, feature order, and configuration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> SoftRoutingLgbmMoe:
    """Load a trusted joblib artifact produced by this project."""
    model = joblib.load(path)
    if not isinstance(model, SoftRoutingLgbmMoe):
        raise TypeError("Artifact does not contain a SoftRoutingLgbmMoe model.")
    return model


def _validate_training_frame(frame: pd.DataFrame, features: list[str]) -> None:
    if frame.empty:
        raise ValueError("Training frame must contain rows.")
    if not features or len(features) != len(set(features)):
        raise ValueError("Feature names must be nonempty and unique.")
    required = ["masked_consumer_id", "FPF_TARGET", *features]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Training frame is missing columns: {missing}")
    if not frame["FPF_TARGET"].isin([0, 1]).all():
        raise ValueError("Training targets must be binary labels 0 and 1 with no missing values.")
    valid_ids = frame["masked_consumer_id"].map(
        lambda value: isinstance(value, str) and len(value) >= 3
    )
    if not valid_ids.all():
        raise ValueError("Consumer IDs must be strings with a three-character client prefix.")
    values = frame[features]
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in values.dtypes):
        raise ValueError("All model features must be numeric.")
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError("Model features may contain missing values but not infinity.")
