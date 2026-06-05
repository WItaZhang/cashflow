from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cashflow_ml.data import per_client_stratified_622_split, resolve_data_paths
from cashflow_ml.evaluations import get_evaluator
from cashflow_ml.features import build_feature_frame, resolve_feature_names
from cashflow_ml.models import build_model
from cashflow_ml.utils import configure_logging, create_run_dir, save_json, set_thread_env


def run_experiment(config: dict, config_path: Path) -> dict:
    seed = int(config["seed"])
    n_jobs = int(config.get("runtime", {}).get("n_jobs", 8))
    set_thread_env(n_jobs)

    run_dir = create_run_dir(config_path, config)
    log = configure_logging(run_dir)
    log.info("=== cashflow experiment: %s ===", config["experiment"]["name"])

    consumer_file, transactions_dir = resolve_data_paths(config)
    feature_sets = config["features"]["feature_sets"]
    feature_cols = resolve_feature_names(feature_sets, config["features"].get("extra_features"))

    log.info("Building features: %s", feature_sets)
    extra_features = config["features"].get("extra_features") or []
    df = build_feature_frame(consumer_file, transactions_dir, feature_sets, extra_features or None)
    labeled = df[df["FPF_TARGET"].notna()].copy()
    log.info("Rows: total=%d labeled=%d features=%d", len(df), len(labeled), len(feature_cols))

    split_cfg = config.get("split", {})
    if split_cfg.get("name", "per_client_stratified_622") != "per_client_stratified_622":
        raise ValueError("Only split.name=per_client_stratified_622 is implemented.")
    train_df, val_df, test_df = per_client_stratified_622_split(labeled, seed=seed)

    model = build_model(config["model"], seed=seed, n_jobs=n_jobs)
    model.fit(train_df, feature_cols)

    evaluator = get_evaluator(config["evaluation"]["name"])
    val_pred = model.predict_proba(val_df)
    test_pred = model.predict_proba(test_df)
    metrics = evaluator(test_df, test_pred)
    metrics["val_group_auc"] = evaluator(val_df, val_pred).get("test_group_auc")
    metrics["n_features"] = len(feature_cols)
    log.info("Metrics: %s", metrics)

    model_path = run_dir / "models" / "model.joblib"
    joblib.dump(model, model_path)
    save_json(metrics, run_dir / "metrics.json")

    pred_df = pd.DataFrame({
        "masked_consumer_id": test_df["masked_consumer_id"].values,
        "predicted_score": np.asarray(test_pred),
        "true_label": test_df["FPF_TARGET"].astype(int).values,
    })
    pred_path = run_dir / "predictions" / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    perm_features = config["evaluation"].get("permutation_features", [])
    if perm_features:
        n_repeats = int(config["evaluation"].get("permutation_n_repeats", 10))
        rng = np.random.default_rng(seed)
        evaluator_fn = get_evaluator(config["evaluation"]["name"])
        baseline_auc = metrics["test_group_auc"]
        perm_results = {}
        for feat in perm_features:
            if feat not in feature_cols:
                log.warning("Permutation feature %s not in feature_cols, skipping.", feat)
                continue
            drops = []
            for _ in range(n_repeats):
                X_shuffled = test_df.copy()
                X_shuffled[feat] = rng.permutation(X_shuffled[feat].values)
                pred_shuffled = model.predict_proba(X_shuffled)
                auc_shuffled = evaluator_fn(X_shuffled, pred_shuffled)["test_group_auc"]
                drops.append(baseline_auc - auc_shuffled)
            perm_results[feat] = {
                "mean_auc_drop": float(np.mean(drops)),
                "std_auc_drop": float(np.std(drops)),
            }
            log.info("Permutation  %-30s  drop=%.4f ± %.4f",
                     feat, perm_results[feat]["mean_auc_drop"], perm_results[feat]["std_auc_drop"])
        save_json(perm_results, run_dir / "permutation_importance.json")
        log.info("Saved permutation importance: %s", run_dir / "permutation_importance.json")
        metrics["permutation_importance"] = perm_results

    log.info("Saved model: %s", model_path)
    log.info("Saved predictions: %s", pred_path)
    log.info("Saved metrics: %s", run_dir / "metrics.json")
    return {"run_dir": str(run_dir), "metrics": metrics}
