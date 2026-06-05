from __future__ import annotations

from cashflow_ml.models.moe_lgbm import SoftRoutingLgbmMoe


def build_model(config: dict, seed: int, n_jobs: int):
    name = config["name"]
    version = config.get("version", "v1")
    key = f"{name}:{version}"
    if key == "soft_routing_lgbm_moe:v15_4":
        return SoftRoutingLgbmMoe(config=config, seed=seed, n_jobs=n_jobs)
    raise ValueError(f"Unknown model version '{key}'.")
