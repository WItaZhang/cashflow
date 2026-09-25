from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from cashflow_ml.models import build_model
from cashflow_ml.models.moe_lgbm import SoftRoutingLgbmMoe
from cashflow_ml.trainer import fit_model, load_model, save_model


class FixedClassifier:
    def __init__(self, classes, probabilities):
        self.classes_ = np.asarray(classes)
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, frame):
        return self.probabilities[: len(frame)]


def mixture():
    model = SoftRoutingLgbmMoe(config={}, seed=42, n_jobs=1)
    model.feature_cols = ["cashflow"]
    model.router_classes = np.array(["C03", "C01", "C02"])
    model.router = FixedClassifier(model.router_classes, [[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    # Reverse one expert's class order to check positive-class alignment.
    model.experts = {
        "C01": FixedClassifier([0, 1], [[0.8, 0.2], [0.6, 0.4]]),
        "C03": FixedClassifier([1, 0], [[0.9, 0.1], [0.5, 0.5]]),
    }
    model.expert_cols = {client: model.feature_cols for client in model.experts}
    return model


def test_soft_routing_aligns_classes_and_renormalizes_missing_experts():
    frame = pd.DataFrame({"cashflow": [10.0, -4.0]})
    scores = mixture().predict_proba(frame)
    np.testing.assert_allclose(
        scores, [(0.5 * 0.9 + 0.3 * 0.2) / 0.8, (0.1 * 0.5 + 0.2 * 0.4) / 0.3]
    )


def test_router_zero_mass_and_missing_experts_raise():
    frame = pd.DataFrame({"cashflow": [10.0]})
    model = mixture()
    model.router.probabilities = np.array([[0.0, 0.0, 1.0]])
    with pytest.raises(RuntimeError, match="zero probability"):
        model.predict_proba(frame)
    model.experts.clear()
    with pytest.raises(RuntimeError, match="No fitted expert"):
        model.predict_proba(frame)


def training_case():
    rng = np.random.default_rng(8)
    frame = pd.DataFrame(
        {
            "masked_consumer_id": [
                f"{client}_{row}" for client in ("C01", "C02", "C03") for row in range(40)
            ],
            "FPF_TARGET": np.tile(np.arange(40) % 4 == 0, 3).astype(int),
            "cashflow": rng.normal(size=120),
            "income": rng.normal(size=120),
        }
    )
    hp = {"n_estimators": 8, "num_leaves": 5, "min_child_samples": 2}
    config = {
        "name": "soft_routing_lgbm_moe",
        "version": "v15_4",
        "hyperparameters": {
            "router": hp,
            "experts": {client: hp for client in ("C01", "C02", "C03")},
        },
        "training": {
            "downsample_negative_clients": ["C01", "C03"],
            "downsample_negative_keep_ratio": 0.2,
            "smote_positive_clients": ["C01", "C03"],
            "smote_positive_multiplier": 2.0,
            "smote_k_neighbors": 5,
            "sample_weight": True,
        },
    }
    return frame, config


def test_fit_and_artifact_roundtrip_preserve_predictions(tmp_path):
    frame, config = training_case()
    original = frame.copy(deep=True)
    model = fit_model(build_model(config, seed=42, n_jobs=1), frame, ["cashflow", "income"])
    scores = model.predict_proba(frame)
    assert set(model.experts) == {"C01", "C02", "C03"}
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    pd.testing.assert_frame_equal(frame, original)
    path = tmp_path / "models" / "model.joblib"
    save_model(model, path)
    np.testing.assert_array_equal(load_model(path).predict_proba(frame), scores)


def test_router_uses_original_rows_and_weights_also_apply_to_unmodified_groups(monkeypatch):
    import cashflow_ml.trainer as trainer

    calls = []

    class RecordingEstimator:
        def fit(self, X, y, **kwargs):
            self.classes_ = np.unique(y)
            self.named_steps = {"clf": self}
            calls.append((X.copy(), np.asarray(y).copy(), kwargs))
            return self

    monkeypatch.setattr(trainer, "build_lgbm", lambda *args: RecordingEstimator())
    frame, config = training_case()
    fit_model(build_model(config, seed=42, n_jobs=1), frame, ["cashflow", "income"])
    assert [len(X) for X, _, _ in calls] == [120, 26, 40, 26]
    np.testing.assert_array_equal(calls[0][1], frame.masked_consumer_id.str[:3])
    assert calls[0][2] == {}
    for _, labels, kwargs in calls[1:]:
        np.testing.assert_array_equal(kwargs["clf__sample_weight"], np.where(labels == 0, 5.0, 0.5))


def test_training_rejects_invalid_labels_and_absent_experts():
    frame, config = training_case()
    invalid = frame.copy()
    invalid["FPF_TARGET"] = invalid["FPF_TARGET"].astype(float)
    invalid.loc[0, "FPF_TARGET"] = 0.5
    with pytest.raises(ValueError, match="binary labels"):
        fit_model(build_model(config, 42, 1), invalid, ["cashflow"])
    config = deepcopy(config)
    config["training"]["min_expert_rows"] = 1000
    with pytest.raises(ValueError, match="No eligible experts"):
        fit_model(build_model(config, 42, 1), frame, ["cashflow"])
