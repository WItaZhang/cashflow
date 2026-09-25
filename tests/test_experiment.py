import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from test_data import demo_config

from cashflow_ml.data import load_inputs
from cashflow_ml.demo import write_demo_inputs
from cashflow_ml.experiment import run_experiment
from cashflow_ml.features import build_feature_frame
from cashflow_ml.trainer import load_model
from cashflow_ml.utils import create_run_dir, run_logging


@pytest.mark.parametrize("evaluator", ["standard_holdout", "auc_only"])
def test_demo_run_persists_replayable_artifacts(tmp_path, evaluator):
    config = demo_config(tmp_path)
    config["evaluation"]["name"] = evaluator
    assert write_demo_inputs(config) == (260, 24960)
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = run_experiment(config, config_path)
    run_dir = Path(result["run_dir"])
    assert json.loads((run_dir / "metrics.json").read_text()) == result["metrics"]
    assert yaml.safe_load((run_dir / "config.yaml").read_text()) == config
    assert (run_dir / "source_config.yaml").read_bytes() == config_path.read_bytes()
    assert result["metrics"]["n_features"] == 56
    assert result["metrics"]["permutation_split"] == "validation"
    assert len(json.loads((run_dir / "permutation_importance.json").read_text())) == 3
    assert len(json.loads((run_dir / "feature_names.json").read_text())) == 56
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["seed"] == config["seed"]
    assert "lightgbm" in metadata["dependencies"]
    membership = pd.read_csv(run_dir / "splits.csv")
    assert membership["masked_consumer_id"].is_unique
    assert membership["split"].value_counts().to_dict() == {
        "train": 156,
        "validation": 52,
        "test": 52,
    }
    consumers, transactions = load_inputs(config)
    features = build_feature_frame(consumers, transactions, ["cashflow_56"])
    model = load_model(run_dir / "models" / "model.joblib")
    for split in ("validation", "test"):
        predictions = pd.read_csv(run_dir / "predictions" / f"{split}_predictions.csv")
        ordered = features.set_index("masked_consumer_id").loc[predictions["masked_consumer_id"]]
        np.testing.assert_allclose(
            model.predict_proba(ordered), predictions["predicted_score"], atol=1e-15
        )
    assert "Fitted expert C01" in (run_dir / "run.log").read_text()


def test_demo_reuses_identical_data_and_refuses_to_overwrite(tmp_path):
    config = demo_config(tmp_path)
    assert write_demo_inputs(config) == write_demo_inputs(config)
    path = Path(config["data"]["consumer_file"])
    before = path.read_bytes()
    config["seed"] += 1
    with pytest.raises(
        (ValueError, FileExistsError), match="(?i)(different|overwrite|existing|match)"
    ):
        write_demo_inputs(config)
    assert path.read_bytes() == before


def test_runtime_config_snapshot_and_collision_isolation(tmp_path):
    config = demo_config(tmp_path)
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump(config), encoding="utf-8")
    config["seed"] = 123
    first, second = create_run_dir(source, config), create_run_dir(source, config)
    assert first != second
    assert yaml.safe_load((first / "config.yaml").read_text())["seed"] == 123
    assert yaml.safe_load((first / "source_config.yaml").read_text())["seed"] == 42


def test_run_log_captures_streams_and_restores_logging(tmp_path):
    previous_handlers = logging.getLogger().handlers[:]
    with run_logging(tmp_path) as log:
        print("stdout marker")
        print("stderr marker", file=sys.stderr)
        log.info("logging marker")
    text = (tmp_path / "run.log").read_text()
    assert all(marker in text for marker in ("stdout marker", "stderr marker", "logging marker"))
    assert logging.getLogger().handlers == previous_handlers
