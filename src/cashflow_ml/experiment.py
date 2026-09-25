"""Wire data, pure features, training, and evaluation into one reproducible run."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cashflow_ml.config import validate_config
from cashflow_ml.data import (
    load_inputs,
    per_client_stratified_622_split,
    resolve_data_paths,
    split_manifest,
    transaction_files,
)
from cashflow_ml.evaluations import get_evaluator, permutation_importance
from cashflow_ml.features import build_feature_frame, resolve_feature_names
from cashflow_ml.models import build_model
from cashflow_ml.trainer import fit_model, save_model
from cashflow_ml.utils import (
    create_run_dir,
    run_logging,
    run_metadata,
    save_json,
    set_thread_env,
)


def _save_predictions(frame: pd.DataFrame, predictions: np.ndarray, path: Path) -> None:
    pd.DataFrame(
        {
            "masked_consumer_id": frame["masked_consumer_id"].values,
            "predicted_score": np.asarray(predictions),
            "true_label": frame["FPF_TARGET"].astype(int).values,
        }
    ).to_csv(path, index=False)


def _evaluate(
    model, validation: pd.DataFrame, test: pd.DataFrame, config: dict, run_dir: Path
) -> dict:
    evaluator = get_evaluator(config["evaluation"]["name"])
    validation_predictions = model.predict_proba(validation)
    test_predictions = model.predict_proba(test)
    metrics = evaluator(test, test_predictions)
    validation_metrics = evaluator(validation, validation_predictions)
    for key, value in validation_metrics.items():
        metrics["val_" + key.removeprefix("test_")] = value
    # Preserve the original key even for evaluators that do not return group AUC.
    metrics.setdefault("val_group_auc", None)
    metrics["n_features"] = len(model.feature_cols)
    _save_predictions(test, test_predictions, run_dir / "predictions" / "test_predictions.csv")
    _save_predictions(
        validation,
        validation_predictions,
        run_dir / "predictions" / "validation_predictions.csv",
    )
    return metrics


def _explain(
    model, validation: pd.DataFrame, test: pd.DataFrame, config: dict, log: logging.Logger
) -> dict:
    evaluation = config["evaluation"]
    requested = evaluation.get("permutation_features", [])
    features = model.feature_cols if requested == "all" else requested
    if not features:
        return {}
    included = []
    for feature in features:
        if feature in model.feature_cols:
            included.append(feature)
        else:
            log.warning("Permutation feature %s not in feature_cols, skipping.", feature)
    split = evaluation.get("permutation_split", "test")
    log.info("Permutation importance: split=%s features=%d", split, len(included))
    importance = permutation_importance(
        model,
        validation if split == "validation" else test,
        included,
        n_repeats=evaluation.get("permutation_n_repeats", 10),
        seed=config["seed"],
        evaluator=evaluation["name"],
        metric=evaluation.get("permutation_metric"),
    )
    for feature, result in importance.items():
        log.info(
            "Permutation %s: AUC drop=%.6f +/- %.6f",
            feature,
            result["mean_auc_drop"],
            result["std_auc_drop"],
        )
    return {"permutation_split": split, "permutation_importance": importance}


def run_experiment(config: dict, config_path: Path) -> dict:
    """Run one configuration. All outputs belong to its unique timestamped folder."""
    validate_config(config)
    seed = config["seed"]
    n_jobs = config.get("runtime", {}).get("n_jobs", 8)
    set_thread_env(n_jobs)
    run_dir = create_run_dir(Path(config_path), config)
    with run_logging(run_dir) as log:
        try:
            log.info("=== cashflow experiment: %s ===", config["experiment"]["name"])
            consumer_path, transactions_path = resolve_data_paths(config)
            save_json(
                run_metadata(config, [consumer_path, *transaction_files(transactions_path)]),
                run_dir / "metadata.json",
            )
            consumers, transactions = load_inputs(config)
            feature_sets = config["features"]["feature_sets"]
            extra = config["features"].get("extra_features")
            feature_cols = resolve_feature_names(feature_sets, extra)
            frame = build_feature_frame(consumers, transactions, feature_sets, extra)
            labeled = frame[frame["FPF_TARGET"].notna()].copy()
            log.info(
                "Rows: total=%d labeled=%d features=%d", len(frame), len(labeled), len(feature_cols)
            )
            save_json(feature_cols, run_dir / "feature_names.json")
            train, validation, test = per_client_stratified_622_split(labeled, seed)
            split_manifest(train, validation, test).to_csv(run_dir / "splits.csv", index=False)
            log.info(
                "Split: train=%d validation=%d test=%d", len(train), len(validation), len(test)
            )
            model = fit_model(build_model(config["model"], seed, n_jobs), train, feature_cols)
            metrics = _evaluate(model, validation, test, config, run_dir)
            explanation = _explain(model, validation, test, config, log)
            metrics.update(explanation)
            if explanation:
                save_json(
                    explanation["permutation_importance"], run_dir / "permutation_importance.json"
                )
            save_model(model, run_dir / "models" / "model.joblib")
            save_json(metrics, run_dir / "metrics.json")
            log.info("Metrics: %s", metrics)
            log.info("Run saved: %s", run_dir)
        except Exception:
            log.exception("Experiment failed; partial outputs are retained at %s", run_dir)
            raise
    return {"run_dir": str(run_dir), "metrics": metrics}
