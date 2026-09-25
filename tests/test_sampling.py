import numpy as np
import pandas as pd
import pytest

from cashflow_ml.sampling import (
    augment_positives,
    downsample_negatives,
    resample_group,
    sample_weights,
)


def test_negative_downsampling_preserves_positives_indices_and_input():
    X = pd.DataFrame({"amount": np.arange(12, dtype=float)}, index=np.arange(20, 32))
    y = pd.Series([0] * 10 + [1, 1], index=X.index)
    original = X.copy(deep=True)
    X_out, y_out = downsample_negatives(X, y, 0.2, np.random.default_rng(42))
    assert y_out.value_counts().to_dict() == {0: 2, 1: 2}
    assert X_out.index.is_monotonic_increasing
    assert X_out.index.equals(y_out.index)
    pd.testing.assert_frame_equal(X_out.loc[[30, 31]], X.loc[[30, 31]])
    pd.testing.assert_frame_equal(X, original)
    repeated, _ = downsample_negatives(X, y, 0.2, np.random.default_rng(42))
    pd.testing.assert_frame_equal(X_out, repeated)


def test_positive_augmentation_interpolates_only_within_positive_class():
    X = pd.DataFrame({"amount": [-100.0, 10.0, 20.0], "other": [np.nan, 2.0, 4.0]})
    y = pd.Series([0, 1, 1])
    X_out, y_out = augment_positives(X, y, 2.0, 5, np.random.default_rng(42))
    assert y_out.tolist() == [0, 1, 1, 1, 1]
    pd.testing.assert_frame_equal(X_out.iloc[:3], X.fillna(0))
    assert X_out.iloc[3:]["amount"].between(10, 20).all()
    np.testing.assert_allclose(X_out.iloc[3:]["other"], X_out.iloc[3:]["amount"] / 5)
    assert pd.isna(X.loc[0, "other"])


def test_unconfigured_group_is_unchanged_and_does_not_consume_randomness():
    X = pd.DataFrame({"amount": [1.0, 2.0, 3.0]})
    y = pd.Series([0, 1, 1])
    rng = np.random.default_rng(42)
    X_out, y_out = resample_group(
        X,
        y,
        "C02",
        {
            "downsample_negative_clients": ["C01"],
            "smote_positive_clients": ["C01"],
        },
        rng,
    )
    pd.testing.assert_frame_equal(X, X_out)
    pd.testing.assert_series_equal(y, y_out)
    assert rng.random() == np.random.default_rng(42).random()


def test_historical_class_weights_are_explicit_and_can_be_disabled():
    y = pd.Series([0, 1, 1, 0])
    training = {
        "sample_weight": True,
        "downsample_negative_keep_ratio": 0.2,
        "smote_positive_multiplier": 2.0,
    }
    np.testing.assert_array_equal(sample_weights(y, training), [5.0, 0.5, 0.5, 5.0])
    assert sample_weights(y, {}) is None


def test_invalid_resampling_parameters_raise():
    X, y = pd.DataFrame({"amount": [1.0, 2.0]}), pd.Series([0, 1])
    with pytest.raises(ValueError, match="keep_ratio"):
        downsample_negatives(X, y, 0, np.random.default_rng(42))
    with pytest.raises(ValueError, match="k_neighbors"):
        augment_positives(X, y, 2, 0, np.random.default_rng(42))
