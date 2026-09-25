"""Stateless training-only resampling, with the original v15_4 RNG sequence.

One generator is shared across clients by the trainer. Client iteration order,
row selection order, rounding, and SMOTE interpolation are intentionally stable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors


def downsample_negatives(
    X: pd.DataFrame,
    y: pd.Series,
    keep_ratio: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.Series]:
    """Retain every positive and a random subset of negatives, in row order."""
    if not 0 < keep_ratio <= 1:
        raise ValueError("downsample_negative_keep_ratio must be in (0, 1].")
    neg_idx = np.where(y.values == 0)[0]
    pos_idx = np.where(y.values == 1)[0]
    n_keep = int(round(len(neg_idx) * keep_ratio))
    keep_neg = rng.choice(neg_idx, size=n_keep, replace=False)
    selected = np.sort(np.concatenate([keep_neg, pos_idx]))
    return X.iloc[selected], y.iloc[selected]


def augment_positives(
    X: pd.DataFrame,
    y: pd.Series,
    positive_multiplier: float,
    k_neighbors: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.Series]:
    """Append positive-class SMOTE samples after zero-imputing this group.

    Nearest neighbors are computed within the positive class only. If fewer
    than two positives exist, interpolation is unavailable and inputs pass
    through unchanged. Originals precede synthetic rows, as in v15_4.
    """
    if not np.isfinite(positive_multiplier) or positive_multiplier < 1:
        raise ValueError("smote_positive_multiplier must be finite and at least 1.")
    if k_neighbors < 1:
        raise ValueError("smote_k_neighbors must be at least 1.")
    n_pos = int((y.values == 1).sum())
    n_synth = max(0, int(round(n_pos * positive_multiplier)) - n_pos)
    if n_synth <= 0 or n_pos < 2:
        return X, y

    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    X_imp = imputer.fit_transform(X)
    X_pos = X_imp[y.values == 1]
    k_eff = min(k_neighbors, n_pos - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_pos)
    _, nbr_idx = nn.kneighbors(X_pos)
    base = rng.integers(0, n_pos, size=n_synth)
    nbr_pick = rng.integers(1, k_eff + 1, size=n_synth)
    chosen = nbr_idx[base, nbr_pick]
    alpha = rng.random(size=(n_synth, 1))
    X_synth = X_pos[base] + alpha * (X_pos[chosen] - X_pos[base])

    original = pd.DataFrame(X_imp, columns=X.columns)
    synth = pd.DataFrame(X_synth, columns=X.columns)
    X_out = pd.concat([original, synth], ignore_index=True)
    y_out = pd.concat(
        [y.reset_index(drop=True), pd.Series([1] * n_synth, dtype=int)],
        ignore_index=True,
    )
    return X_out, y_out


def resample_group(
    X: pd.DataFrame,
    y: pd.Series,
    client: str,
    training: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply configured negative downsampling, then positive augmentation."""
    if client in training.get("downsample_negative_clients", []):
        X, y = downsample_negatives(
            X,
            y,
            training.get("downsample_negative_keep_ratio", 0.2),
            rng,
        )
    if client in training.get("smote_positive_clients", []):
        X, y = augment_positives(
            X,
            y,
            training.get("smote_positive_multiplier", 2.0),
            training.get("smote_k_neighbors", 5),
            rng,
        )
    return X, y


def sample_weights(y: pd.Series, training: dict) -> np.ndarray | None:
    """Return historical class weights for *every* expert when enabled.

    v15_4 uses inverse keep ratio for negatives and inverse augmentation
    multiplier for positives, even for groups that were not resampled. This
    preserves the published implementation; these are class weights, not a
    claim of unbiased sampling correction for unmodified groups.
    """
    if not training.get("sample_weight", False):
        return None
    keep_ratio = training.get("downsample_negative_keep_ratio", 1.0)
    multiplier = training.get("smote_positive_multiplier", 1.0)
    if not 0 < keep_ratio <= 1 or not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError("Sample weights require a keep ratio in (0, 1] and multiplier >= 1.")
    return np.where(y.values == 0, 1.0 / keep_ratio, 1.0 / multiplier).astype(float)
