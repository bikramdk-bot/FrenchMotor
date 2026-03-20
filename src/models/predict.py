"""Batch prediction utilities."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import pandas as pd

from src.models.train import score_models


def load_artifacts(model_path: str | Path) -> Dict:
    """Load serialized model artifacts."""
    with open(model_path, "rb") as file:
        return pickle.load(file)


def predict(df: pd.DataFrame, model_path: str | Path) -> pd.DataFrame:
    """Generate MTPL predictions from persisted artifacts."""
    artifacts = load_artifacts(model_path)
    return score_models(df, artifacts["frequency_model"], artifacts["severity_model"])
