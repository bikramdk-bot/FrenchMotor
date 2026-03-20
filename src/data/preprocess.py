"""Preprocessing steps for MTPL source data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "driver_age",
    "vehicle_age",
    "vehicle_power",
    "bonus_malus",
    "density",
    "exposure",
    "claim_count",
    "claim_amount",
]

RAW_COLUMN_RENAMES = {
    "idpol": "policy_id",
    "claimnb": "claim_count",
    "claimamount": "claim_amount",
    "exposure": "exposure",
    "area": "area",
    "vehpower": "vehicle_power",
    "vehage": "vehicle_age",
    "drivage": "driver_age",
    "bonusmalus": "bonus_malus",
    "vehbrand": "vehicle_brand",
    "vehgas": "fuel_type",
    "density": "density",
    "region": "region",
}


def standardize_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize source column names to a consistent snake_case schema."""
    standardized = df.copy()
    standardized.columns = [column.strip().lower() for column in standardized.columns]
    standardized = standardized.rename(columns=RAW_COLUMN_RENAMES)
    return standardized


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw MTPL data and create modeling targets."""
    cleaned = standardize_source_columns(df)

    missing = [column for column in REQUIRED_COLUMNS if column not in cleaned.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned["exposure"] = cleaned["exposure"].clip(lower=1e-3)
    cleaned["claim_count"] = cleaned["claim_count"].clip(lower=0).astype(int)
    cleaned["claim_amount"] = cleaned["claim_amount"].clip(lower=0.0)
    
    # Create modeling targets
    cleaned["frequency_target"] = cleaned["claim_count"] / cleaned["exposure"]
    
    positive_claims = cleaned["claim_count"] > 0
    cleaned["severity_target"] = 0.0
    cleaned.loc[positive_claims, "severity_target"] = (
        cleaned.loc[positive_claims, "claim_amount"] / cleaned.loc[positive_claims, "claim_count"]
    )
    
    cleaned["pure_premium_target"] = cleaned["claim_amount"] / cleaned["exposure"]

    numeric_columns = cleaned.select_dtypes(include=[np.number]).columns
    cleaned[numeric_columns] = cleaned[numeric_columns].replace([np.inf, -np.inf], np.nan)
    cleaned[numeric_columns] = cleaned[numeric_columns].fillna(cleaned[numeric_columns].median())

    categorical_columns = cleaned.select_dtypes(exclude=[np.number]).columns
    for column in categorical_columns:
        cleaned[column] = cleaned[column].fillna("unknown").astype(str)

    return cleaned


def save_model_input_data(df: pd.DataFrame, config: Dict, filename: str = "mtpl_model_input.csv") -> Path:
    """Persist model-ready data to the configured model input directory."""
    model_input_dir = Path(config["paths"]["model_input_dir"])
    model_input_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_input_dir / filename
    df.to_csv(output_path, index=False)
    return output_path


def save_processed_data(df: pd.DataFrame, config: Dict, filename: str = "mtpl_model_input.csv") -> Path:
    """Backward-compatible wrapper for saving model-ready data."""
    return save_model_input_data(df, config, filename=filename)
