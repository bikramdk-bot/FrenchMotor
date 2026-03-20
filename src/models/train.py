"""Model training routines for MTPL frequency and severity."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features.feature_engineering import build_preprocessor


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_models(df: pd.DataFrame, config: Dict) -> Dict:
    """Train baseline frequency and severity models and return artifacts."""
    train_df, test_df = train_test_split(
        df,
        test_size=config["training"].get("test_size", 0.2),
        random_state=config["training"].get("random_state", 42),
    )

    preprocessor = build_preprocessor(train_df)

    frequency_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                PoissonRegressor(
                    alpha=config["training"].get("poisson_alpha", 1.0),
                    max_iter=config["training"].get("max_iter", 300),
                ),
            ),
        ]
    )
    frequency_model.fit(train_df, train_df["claim_count"], model__sample_weight=train_df["exposure"])

    severity_model = None
    severity_train = train_df[(train_df["claim_count"] > 0) & (train_df["severity_target"] > 0)].copy()
    if not severity_train.empty and float(severity_train["claim_amount"].sum()) > 0.0:
        severity_preprocessor = build_preprocessor(severity_train)
        severity_model = Pipeline(
            steps=[
                ("preprocessor", severity_preprocessor),
                (
                    "model",
                    GammaRegressor(
                        alpha=config["training"].get("gamma_alpha", 0.5),
                        max_iter=config["training"].get("max_iter", 300),
                    ),
                ),
            ]
        )
        severity_model.fit(severity_train, severity_train["severity_target"])

    test_predictions = score_models(test_df, frequency_model, severity_model)

    metrics = {
        "frequency_mae": float(mean_absolute_error(test_df["frequency_target"], test_predictions["predicted_frequency"])),
    }
    if severity_model is not None:
        metrics["severity_mae"] = float(mean_absolute_error(test_df["severity_target"], test_predictions["predicted_severity"]))
        metrics["pure_premium_mae"] = float(
            mean_absolute_error(test_df["pure_premium_target"], test_predictions["predicted_pure_premium"])
        )
        metrics["pure_premium_rmse"] = _rmse(test_df["pure_premium_target"], test_predictions["predicted_pure_premium"])

    return {
        "frequency_model": frequency_model,
        "severity_model": severity_model,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "metrics": metrics,
        "reference_data": train_df.reset_index(drop=True),
        "evaluation_data": test_predictions.reset_index(drop=True),
    }


def score_models(df: pd.DataFrame, frequency_model: Pipeline, severity_model: Pipeline | None) -> pd.DataFrame:
    """Run model scoring and derive pure premium predictions."""
    scored = df.copy()
    predicted_claim_count = frequency_model.predict(scored).clip(min=0.0)
    predicted_frequency = predicted_claim_count / scored["exposure"].clip(lower=1e-3)
    if severity_model is None:
        predicted_severity = np.zeros(len(scored), dtype=float)
    else:
        predicted_severity = severity_model.predict(scored).clip(min=0.0)

    scored["predicted_claim_count"] = predicted_claim_count
    scored["predicted_frequency"] = predicted_frequency
    scored["predicted_severity"] = predicted_severity
    scored["predicted_pure_premium"] = predicted_frequency * predicted_severity
    return scored


def save_artifacts(artifacts: Dict, config: Dict, filename: str = "baseline_model.pkl") -> Path:
    """Serialize model artifacts to disk."""
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = models_dir / filename

    payload = {
        "frequency_model": artifacts["frequency_model"],
        "severity_model": artifacts["severity_model"],
        "metrics": artifacts["metrics"],
        "train_rows": artifacts["train_rows"],
        "test_rows": artifacts["test_rows"],
        "has_severity_model": artifacts["severity_model"] is not None,
    }
    with open(output_path, "wb") as file:
        pickle.dump(payload, file)
    return output_path
