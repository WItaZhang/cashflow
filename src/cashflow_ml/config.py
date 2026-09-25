"""Explicit validation of the experiment configuration contract."""

from __future__ import annotations

import math
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("Experiment YAML must contain a mapping.")
    required = {
        "experiment": ("name",),
        "data": ("consumer_file", "transactions_dir"),
        "features": ("feature_sets",),
        "model": ("name", "hyperparameters"),
        "evaluation": ("name",),
    }
    for section, keys in required.items():
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Config section '{section}' must be a mapping.")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing config value: {section}.{key}")
    for section in ("runtime", "split"):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"Config section '{section}' must be a mapping.")
    if type(config.get("seed")) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    name = config["experiment"]["name"]
    if (
        not isinstance(name, str)
        or not name
        or any(char in name for char in ("/", chr(92), ":"))
        or name in {".", ".."}
    ):
        raise ValueError("experiment.name must be a directory name without path separators.")
    n_jobs = config.get("runtime", {}).get("n_jobs", 8)
    if type(n_jobs) is not int or n_jobs < 1:
        raise ValueError("runtime.n_jobs must be a positive integer.")
    sets = config["features"]["feature_sets"]
    if not isinstance(sets, list) or not sets or not all(isinstance(name, str) for name in sets):
        raise ValueError("features.feature_sets must be a nonempty list of names.")
    extra = config["features"].get("extra_features", [])
    if extra is not None and (
        not isinstance(extra, list) or not all(isinstance(name, str) for name in extra)
    ):
        raise ValueError("features.extra_features must be a list of names.")
    if (
        config.get("split", {}).get("name", "per_client_stratified_622")
        != "per_client_stratified_622"
    ):
        raise ValueError("Only split.name=per_client_stratified_622 is implemented.")
    model = config["model"]
    hp = model["hyperparameters"]
    if (
        not isinstance(hp, dict)
        or not isinstance(hp.get("router"), dict)
        or not isinstance(hp.get("experts"), dict)
        or not hp["experts"]
    ):
        raise ValueError("model.hyperparameters requires router and nonempty experts mappings.")
    training = model.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("model.training must be a mapping.")
    for key, default, lower, upper in (
        ("downsample_negative_keep_ratio", 0.2, 0, 1),
        ("smote_positive_multiplier", 2.0, 1, math.inf),
    ):
        value = training.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < lower
            or value > upper
            or value == 0
        ):
            raise ValueError(f"model.training.{key} is out of range.")
    for key, default in (("min_expert_rows", 20), ("smote_k_neighbors", 5)):
        if type(training.get(key, default)) is not int or training.get(key, default) < 1:
            raise ValueError(f"model.training.{key} must be a positive integer.")
    for key in ("downsample_negative_clients", "smote_positive_clients"):
        clients = training.get(key, [])
        if not isinstance(clients, list) or any(client not in hp["experts"] for client in clients):
            raise ValueError(f"model.training.{key} must list configured experts.")
    if "sample_weight" in training and type(training["sample_weight"]) is not bool:
        raise ValueError("model.training.sample_weight must be a boolean.")
    evaluation = config["evaluation"]
    if evaluation.get("permutation_split", "test") not in {"validation", "test"}:
        raise ValueError("evaluation.permutation_split must be validation or test.")
    features = evaluation.get("permutation_features", [])
    if features != "all" and (
        not isinstance(features, list) or not all(isinstance(name, str) for name in features)
    ):
        raise ValueError("evaluation.permutation_features must be a list of names or 'all'.")
    repeats = evaluation.get("permutation_n_repeats", 10)
    if type(repeats) is not int or repeats < 1:
        raise ValueError("evaluation.permutation_n_repeats must be a positive integer.")
