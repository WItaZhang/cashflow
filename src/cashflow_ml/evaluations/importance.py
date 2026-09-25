"""Reproducible permutation importance on a caller-selected holdout frame."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from cashflow_ml.evaluations.registry import get_evaluator


def permutation_importance(
    model: object,
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    n_repeats: int,
    seed: int,
    evaluator: str | Callable[[pd.DataFrame, object], dict] = "standard_holdout",
    metric: str | None = None,
) -> dict[str, dict[str, float]]:
    """Measure score drops without fitting the model or mutating the input.

    The evaluator must expose a scalar, higher-is-better metric. Historical
    output keys and population standard deviation (``ddof=0``) are retained.
    The caller chooses validation or test data explicitly.
    """
    if n_repeats < 1:
        raise ValueError("Permutation importance requires n_repeats >= 1.")
    if len(features) != len(set(features)):
        raise ValueError("Permutation features must be unique.")
    unknown = [feature for feature in features if feature not in model.feature_cols]
    if unknown:
        raise ValueError(f"Permutation features are not model inputs: {unknown}")
    evaluate = get_evaluator(evaluator) if isinstance(evaluator, str) else evaluator
    baseline_metrics = evaluate(frame, model.predict_proba(frame))
    if metric is None:
        metric = "test_group_auc" if "test_group_auc" in baseline_metrics else "test_auc"
    if metric not in baseline_metrics:
        raise ValueError(f"Evaluator does not produce permutation metric {metric!r}.")
    baseline = float(baseline_metrics[metric])
    if not np.isfinite(baseline):
        raise ValueError(f"Permutation metric {metric!r} must be finite on the evaluation frame.")
    rng = np.random.default_rng(seed)
    results = {}
    for feature in features:
        drops = []
        for _ in range(n_repeats):
            shuffled = frame.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].values)
            score = float(evaluate(shuffled, model.predict_proba(shuffled))[metric])
            drops.append(baseline - score)
        results[feature] = {
            "mean_auc_drop": float(np.mean(drops)),
            "std_auc_drop": float(np.std(drops)),
        }
    return results
