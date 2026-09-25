"""Run directories, logging, JSON artifacts, and reproducibility metadata."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import TextIO

import yaml

from cashflow_ml.config import load_config as load_config


def save_json(obj: object, path: Path) -> None:
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_thread_env(n_threads: int) -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "LIGHTGBM_NUM_THREADS",
    ):
        os.environ[key] = str(n_threads)


def create_run_dir(config_path: Path, config: dict) -> Path:
    root = Path(config["experiment"].get("log_root", "logs"))
    name = f"{timestamp()}_{config['experiment']['name']}"
    # Concurrent or rapid repeated runs must not share an artifact directory.
    for attempt in range(1000):
        run_dir = root / (name if attempt == 0 else f"{name}_{attempt:03d}")
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError(f"Cannot reserve a unique run directory under {root}")
    shutil.copy2(config_path, run_dir / "source_config.yaml")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "models").mkdir()
    (run_dir / "predictions").mkdir()
    return run_dir


class _Tee:
    def __init__(self, console: TextIO, log_file: TextIO):
        self.console = console
        self.log_file = log_file

    def write(self, message: str) -> int:
        self.console.write(message)
        self.log_file.write(message)
        return len(message)

    def flush(self) -> None:
        self.console.flush()
        self.log_file.flush()


@contextmanager
def run_logging(run_dir: Path) -> Iterator[logging.Logger]:
    """Capture Python stdout/stderr and logging for this run, then restore them."""
    root = logging.getLogger()
    previous_handlers, previous_level = root.handlers[:], root.level
    with (run_dir / "run.log").open("w", encoding="utf-8") as stream:
        with redirect_stdout(_Tee(sys.stdout, stream)), redirect_stderr(_Tee(sys.stderr, stream)):
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            root.handlers = [handler]
            root.setLevel(logging.INFO)
            try:
                yield logging.getLogger("cashflow_ml")
            finally:
                handler.flush()
                root.handlers = previous_handlers
                root.setLevel(previous_level)
                handler.close()


def run_metadata(config: dict, input_files: list[Path]) -> dict:
    """Record source/environment and inexpensive input identities; no data upload."""
    repository = Path(__file__).resolve().parents[2]
    source = {"commit": None, "dirty": None}
    try:
        source["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        source["dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repository,
                text=True,
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "source": source,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": config["seed"],
        "dependencies": {
            name: version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "lightgbm",
                "pyarrow",
                "joblib",
            )
        },
        "inputs": [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "modified_ns": path.stat().st_mtime_ns,
            }
            for path in input_files
        ],
    }
