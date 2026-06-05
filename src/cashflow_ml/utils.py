from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_thread_env(n_threads: int) -> None:
    value = str(n_threads)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "LIGHTGBM_NUM_THREADS",
    ):
        os.environ[key] = value


def create_run_dir(config_path: Path, config: dict) -> Path:
    exp_name = config["experiment"]["name"]
    root = Path(config["experiment"].get("log_root", "logs"))
    run_dir = root / f"{timestamp()}_{exp_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "config.yaml")
    (run_dir / "models").mkdir()
    (run_dir / "predictions").mkdir()
    return run_dir


def configure_logging(run_dir: Path) -> logging.Logger:
    log_path = run_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger("cashflow_ml")
