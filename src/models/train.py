"""Model training routines for MTPL frequency and severity."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline

import xgboost as xgb
from xgboost import XGBRegressor

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

    use_xgb = bool(config["training"].get("use_xgboost", False))
    n_splits = int(config["training"].get("n_splits", 5))

    # Frequency model: either Poisson GLM or XGBoost with Poisson objective
    if use_xgb:
        # If a tuned artifact exists, reuse its best params to train on full data
        models_dir = Path(config["paths"]["models_dir"])
        tuned_path = models_dir / "xgb_freq_tuned.pkl"
        if tuned_path.exists():
            try:
                with open(tuned_path, "rb") as f:
                    tuned = pickle.load(f)
                best_est = tuned.get("best_estimator") if isinstance(tuned, dict) else None
                if best_est is not None:
                    try:
                        best_model_obj = best_est.named_steps["model"]
                    except Exception:
                        best_model_obj = best_est
                    best_params = best_model_obj.get_params()
                    xgb_model = XGBRegressor(objective="count:poisson", n_jobs=1, **best_params)
                else:
                    xgb_model = XGBRegressor(objective="count:poisson", n_jobs=1, **config["training"].get("xgb_freq_params", {}))
            except Exception:
                xgb_model = XGBRegressor(objective="count:poisson", n_jobs=1, **config["training"].get("xgb_freq_params", {}))
        else:
            xgb_model = XGBRegressor(objective="count:poisson", n_jobs=1, **config["training"].get("xgb_freq_params", {}))

        freq_model = Pipeline(steps=[("preprocessor", preprocessor), ("model", xgb_model)])
        freq_model.fit(train_df, train_df["claim_count"], model__sample_weight=train_df["exposure"])
        frequency_model = freq_model
    else:
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

    # Severity: train both GLM (Gamma) and XGBoost regressor (tree) and compare via K-fold CV
    severity_model = None
    severity_train = train_df[(train_df["claim_count"] > 0) & (train_df["severity_target"] > 0)].copy()
    if not severity_train.empty and float(severity_train["claim_amount"].sum()) > 0.0:
        # prepare pipelines
        severity_preprocessor = build_preprocessor(severity_train)
        glm_sev = Pipeline(
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

        xgb_sev = Pipeline(
            steps=[
                ("preprocessor", severity_preprocessor),
                (
                    "model",
                    XGBRegressor(objective="reg:squarederror", n_jobs=1, **config["training"].get("xgb_sev_params", {})),
                ),
            ]
        )

        # K-fold CV to compare severity models
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=config["training"].get("random_state", 42))
        glm_cv_scores = []
        xgb_cv_scores = []
        for train_idx, val_idx in kf.split(severity_train):
            tr = severity_train.iloc[train_idx]
            va = severity_train.iloc[val_idx]

            glm_sev.fit(tr, tr["severity_target"])
            pred_glm = glm_sev.predict(va).clip(min=0.0)
            glm_cv_scores.append(float(mean_absolute_error(va["severity_target"], pred_glm)))

            xgb_sev.fit(tr, tr["severity_target"])
            pred_xgb = xgb_sev.predict(va).clip(min=0.0)
            xgb_cv_scores.append(float(mean_absolute_error(va["severity_target"], pred_xgb)))

        glm_mean_mae = float(np.mean(glm_cv_scores)) if glm_cv_scores else float("nan")
        xgb_mean_mae = float(np.mean(xgb_cv_scores)) if xgb_cv_scores else float("nan")

        # choose winner (lower MAE)
        if xgb_mean_mae < glm_mean_mae:
            severity_model = xgb_sev
            # fit on full severity_train
            severity_model.fit(severity_train, severity_train["severity_target"])
            selected_severity = "xgboost"
        else:
            severity_model = glm_sev
            severity_model.fit(severity_train, severity_train["severity_target"])
            selected_severity = "glm_gamma"
    else:
        glm_mean_mae = xgb_mean_mae = float("nan")
        selected_severity = None

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
