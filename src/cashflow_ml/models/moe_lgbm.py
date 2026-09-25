"""Fitted mixture state and soft-routing inference.

The router estimates client membership from cashflow features. Each expert
estimates credit risk; its prediction is weighted by the router's probability
for that client. Training and resampling live in ``cashflow_ml.trainer``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SoftRoutingLgbmMoe:
    """A learned LightGBM router and one binary expert per eligible client.

    ``predict_proba`` returns a one-dimensional positive-class risk score,
    preserving the project's original API. Missing experts are excluded and
    the remaining routing probabilities are renormalized per row.
    """

    config: dict
    seed: int
    n_jobs: int
    router: object | None = None
    router_classes: np.ndarray | None = None
    experts: dict[str, object] = field(default_factory=dict)
    expert_cols: dict[str, list[str]] = field(default_factory=dict)
    feature_cols: list[str] = field(default_factory=list)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Blend expert risks using the router's learned class ordering."""
        if self.router is None or self.router_classes is None:
            raise RuntimeError("Model is not fitted; call trainer.fit_model first.")
        if not any(cid in self.experts for cid in self.router_classes):
            raise RuntimeError("No fitted expert matches a router class.")
        missing = [column for column in self.feature_cols if column not in df.columns]
        if missing:
            raise ValueError(f"Prediction frame is missing features: {missing}")
        X = df[self.feature_cols]
        if X.empty:
            return np.empty(0, dtype=float)

        router_probs = _probabilities(self.router, X, len(self.router_classes))
        out = np.zeros(len(X))
        total_w = np.zeros(len(X))
        for j, cid in enumerate(self.router_classes):
            if cid not in self.experts:
                continue
            expert = self.experts[cid]
            classes = np.asarray(expert.classes_)
            positive_columns = np.flatnonzero(classes == 1)
            if len(positive_columns) != 1 or set(classes) != {0, 1}:
                raise RuntimeError(f"Expert {cid!r} must be fitted on binary labels 0 and 1.")
            expert_probs = _probabilities(expert, X[self.expert_cols[cid]], len(classes))
            p_expert = expert_probs[:, positive_columns[0]]
            out += router_probs[:, j] * p_expert
            total_w += router_probs[:, j]
        if np.any(total_w <= 0):
            raise RuntimeError("Router assigned zero probability to every available expert.")
        # Preserve the historical denominator floor for numerical equivalence.
        return out / np.maximum(total_w, 1e-9)


def _probabilities(estimator: object, X: pd.DataFrame, n_classes: int) -> np.ndarray:
    values = np.asarray(estimator.predict_proba(X), dtype=float)
    if values.shape != (len(X), n_classes):
        raise RuntimeError("Estimator returned probabilities with an unexpected shape.")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise RuntimeError("Estimator returned invalid probabilities.")
    return values
