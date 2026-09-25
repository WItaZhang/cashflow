from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from cashflow_ml.config import load_config, validate_config
from cashflow_ml.data import load_inputs, per_client_stratified_622_split
from cashflow_ml.demo import write_demo_inputs

ROOT = Path(__file__).resolve().parents[1]


def demo_config(tmp_path):
    config = load_config(ROOT / "configs" / "demo.yaml")
    config["demo"]["staging_root"] = str(tmp_path)
    config["data"] = {
        "consumer_file": str(tmp_path / "inputs" / "consumer_data.parquet"),
        "transactions_dir": str(tmp_path / "inputs" / "transactions"),
    }
    config["experiment"]["log_root"] = str(tmp_path / "logs")
    return config


def test_split_is_reproducible_disjoint_and_complete():
    frame = pd.DataFrame(
        {
            "masked_consumer_id": [
                f"{client}_{i}" for client in ("C01", "C02", "C03") for i in range(100)
            ],
            "FPF_TARGET": [i % 5 == 0 for _ in range(3) for i in range(100)],
        }
    )
    split = per_client_stratified_622_split(frame, seed=42)
    repeated = per_client_stratified_622_split(frame, seed=42)
    ids = [set(part["masked_consumer_id"]) for part in split]
    assert [len(part) for part in split] == [180, 60, 60]
    assert set.union(*ids) == set(frame["masked_consumer_id"])
    assert not any(ids[i] & ids[j] for i in range(3) for j in range(i))
    for part, second in zip(split, repeated):
        pd.testing.assert_frame_equal(part, second)
        assert set(part["masked_consumer_id"].str[:3]) == {"C01", "C02", "C03"}
        assert part["FPF_TARGET"].mean() == pytest.approx(0.2)


def test_duplicate_consumers_are_rejected_before_split():
    frame = pd.DataFrame({"masked_consumer_id": ["C01_a", "C01_a"], "FPF_TARGET": [0, 1]})
    with pytest.raises(ValueError, match="one row per consumer"):
        per_client_stratified_622_split(frame, 42)


def test_missing_transaction_shards_have_actionable_error(tmp_path):
    config = demo_config(tmp_path)
    consumer_file = Path(config["data"]["consumer_file"])
    consumer_file.parent.mkdir(parents=True)
    consumer_file.touch()
    Path(config["data"]["transactions_dir"]).mkdir()
    with pytest.raises(FileNotFoundError, match="No transactions_"):
        load_inputs(config)


def test_loader_validates_consumer_schema(tmp_path):
    config = demo_config(tmp_path)
    write_demo_inputs(config)
    consumer_file = Path(config["data"]["consumer_file"])
    consumers = pd.read_parquet(consumer_file).drop(columns="FPF_TARGET")
    consumers.to_parquet(consumer_file, index=False)
    with pytest.raises(ValueError, match="missing required columns.*FPF_TARGET"):
        load_inputs(config)


@pytest.mark.parametrize(
    "path", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda path: path.name
)
def test_shipped_configs_are_valid(path):
    validate_config(load_config(path))


@pytest.mark.parametrize(
    "key,value", [("downsample_negative_keep_ratio", 0), ("smote_positive_multiplier", 0.5)]
)
def test_invalid_sampling_config_fails_before_training(key, value):
    config = deepcopy(load_config(ROOT / "configs" / "cashflow_moe.yaml"))
    config["model"]["training"][key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(config)
