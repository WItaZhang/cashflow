import numpy as np
import pandas as pd
import pytest

from cashflow_ml.evaluations import get_evaluator, permutation_importance
from cashflow_ml.evaluations.metrics import group_auc, per_group_auc


def test_group_auc_is_equal_weighted_and_skips_single_class_groups():
    frame = pd.DataFrame(
        {
            "masked_consumer_id": ["C01_a", "C01_b", "C02_a", "C02_b", "C02_c", "C02_d", "C03_a"],
            "FPF_TARGET": [0, 1, 0, 0, 1, 1, 0],
        }
    )
    scores = [0.1, 0.9, 0.9, 0.8, 0.2, 0.1, 0.5]
    assert per_group_auc(frame, scores) == {"C01": 1.0, "C02": 0.0}
    assert group_auc(frame, scores) == 0.5
    metrics = get_evaluator("standard_holdout")(frame, scores)
    assert set(metrics) == {"test_auc", "test_group_auc", "per_client_auc"}


class FeatureScore:
    feature_cols = ["signal", "constant"]

    def predict_proba(self, frame):
        return frame["signal"].to_numpy()


def test_permutation_importance_is_reproducible_and_does_not_mutate_holdout():
    labels = np.tile([0, 1], 20)
    frame = pd.DataFrame(
        {
            "masked_consumer_id": [f"C01_{i}" for i in range(len(labels))],
            "FPF_TARGET": labels,
            "signal": labels.astype(float),
            "constant": np.ones(len(labels)),
        }
    )
    original = frame.copy(deep=True)
    kwargs = dict(n_repeats=5, seed=42, evaluator="auc_only", metric="test_auc")
    results = permutation_importance(FeatureScore(), frame, ["signal", "constant"], **kwargs)
    assert results == permutation_importance(
        FeatureScore(), frame, ["signal", "constant"], **kwargs
    )
    assert results["signal"]["mean_auc_drop"] > 0.3
    assert results["constant"] == {"mean_auc_drop": 0.0, "std_auc_drop": 0.0}
    pd.testing.assert_frame_equal(frame, original)


def test_permutation_importance_rejects_unknown_features():
    with pytest.raises(ValueError, match="not model inputs"):
        permutation_importance(FeatureScore(), pd.DataFrame(), ["unknown"], n_repeats=1, seed=42)


@pytest.mark.parametrize("evaluator", ["standard_holdout", "auc_only"])
def test_undefined_auc_fails_instead_of_producing_nonstandard_json(evaluator):
    frame = pd.DataFrame({"masked_consumer_id": ["C01_a", "C01_b"], "FPF_TARGET": [0, 0]})
    with pytest.raises(ValueError, match="undefined"):
        get_evaluator(evaluator)(frame, [0.1, 0.2])
