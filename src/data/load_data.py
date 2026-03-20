"""Data loading utilities for MTPL-style datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str | Path) -> Dict:
    """Load YAML configuration from disk."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge config dictionaries with override precedence."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_project_config(project_root: str | Path, profile: str | None = None) -> Dict:
    """Load base, validation, and optional profile config into one runtime config."""
    project_root = Path(project_root)
    config_dir = project_root / "config"

    base_config = load_config(config_dir / "config.yaml")
    validation_path = config_dir / "data_validation.yaml"
    if validation_path.exists():
        base_config = _deep_merge(base_config, load_config(validation_path))

    if profile:
        profile_path = config_dir / "profiles" / f"{profile}.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile config not found: {profile_path}")
        base_config = _deep_merge(base_config, load_config(profile_path))

    return base_config


def _generate_synthetic_mtpl(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic MTPL-like portfolio when no raw CSV is available."""
    rng = np.random.default_rng(random_state)

    driver_age = rng.integers(18, 85, size=n_samples)
    vehicle_age = rng.integers(0, 20, size=n_samples)
    vehicle_power = rng.normal(95, 25, size=n_samples).clip(40, 220)
    bonus_malus = rng.normal(1.0, 0.25, size=n_samples).clip(0.5, 2.0)
    density = rng.lognormal(mean=4.8, sigma=0.55, size=n_samples)
    exposure = rng.uniform(0.4, 1.0, size=n_samples)
    region = rng.choice(["north", "south", "east", "west"], size=n_samples, p=[0.2, 0.3, 0.25, 0.25])
    fuel_type = rng.choice(["petrol", "diesel", "hybrid", "electric"], size=n_samples, p=[0.45, 0.35, 0.15, 0.05])

    region_factor = pd.Series(region).map({"north": 0.95, "south": 1.10, "east": 1.05, "west": 0.90}).to_numpy()
    fuel_factor = pd.Series(fuel_type).map({"petrol": 1.0, "diesel": 1.05, "hybrid": 0.92, "electric": 0.88}).to_numpy()

    log_lambda = (
        -2.8
        + 0.012 * np.maximum(vehicle_power - 100, 0)
        + 0.015 * np.maximum(25 - driver_age, 0)
        + 0.035 * vehicle_age
        + 0.4 * (bonus_malus - 1.0)
        + 0.00008 * density
        + np.log(exposure)
        + np.log(region_factor)
        + np.log(fuel_factor)
    )
    claim_count = rng.poisson(np.exp(log_lambda))

    mean_severity = (
        1200
        + 3.0 * vehicle_power
        + 15.0 * density / 100
        + 120.0 * vehicle_age
        + 250.0 * (bonus_malus - 1.0)
    )
    severity_noise = rng.gamma(shape=2.5, scale=np.maximum(mean_severity, 200) / 2.5)
    claim_amount = np.where(claim_count > 0, claim_count * severity_noise, 0.0)

    return pd.DataFrame(
        {
            "policy_id": np.arange(1, n_samples + 1),
            "driver_age": driver_age,
            "vehicle_age": vehicle_age,
            "vehicle_power": vehicle_power.round(1),
            "bonus_malus": bonus_malus.round(3),
            "density": density.round(2),
            "exposure": exposure.round(3),
            "region": region,
            "fuel_type": fuel_type,
            "claim_count": claim_count,
            "claim_amount": claim_amount.round(2),
        }
    )


def load_raw_datasets(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Load the raw frequency and severity datasets from disk."""
    raw_dir = Path(config["paths"]["raw_data_dir"])
    frequency_path = raw_dir / "freMTPL2freq.csv"
    severity_path = raw_dir / "freMTPL2sev.csv"

    if not frequency_path.exists() or not severity_path.exists():
        raise FileNotFoundError(
            "Expected raw MTPL source files 'freMTPL2freq.csv' and 'freMTPL2sev.csv' under data/raw."
        )

    frequency_df = pd.read_csv(frequency_path)
    severity_df = pd.read_csv(severity_path)
    return frequency_df, severity_df, frequency_path, severity_path


def build_claim_level_dataset(freq_df: pd.DataFrame, sev_df: pd.DataFrame) -> pd.DataFrame:
    """Create a claim-level dataset by attaching policy attributes to each claim record."""
    claim_level = sev_df.merge(freq_df, on="IDpol", how="left", validate="many_to_one")
    claim_level.insert(0, "claim_row_id", np.arange(1, len(claim_level) + 1))
    return claim_level


def build_policy_level_dataset(freq_df: pd.DataFrame, sev_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate claim amounts to policy level and join them back to the frequency table."""
    severity_by_policy = (
        sev_df.groupby("IDpol", as_index=False)
        .agg(claim_amount=("ClaimAmount", "sum"))
        .astype({"claim_amount": float})
    )

    policy_level = freq_df.merge(severity_by_policy, on="IDpol", how="left", validate="one_to_one")
    policy_level["claim_amount"] = policy_level["claim_amount"].fillna(0.0)
    return policy_level


def load_raw_data(config: Dict) -> Tuple[pd.DataFrame, Path]:
    """Load the policy-level modeling base table or generate synthetic data."""
    try:
        freq_df, sev_df, frequency_path, _ = load_raw_datasets(config)
        return build_policy_level_dataset(freq_df, sev_df), frequency_path
    except FileNotFoundError:
        pass

    raw_dir = Path(config["paths"]["raw_data_dir"])
    synthetic = _generate_synthetic_mtpl(
        n_samples=config["data"].get("synthetic_rows", 5000),
        random_state=config["data"].get("random_state", 42),
    )
    output_path = raw_dir / "synthetic_mtpl.csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(output_path, index=False)
    return synthetic, output_path
