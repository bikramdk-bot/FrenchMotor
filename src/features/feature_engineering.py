"""Feature selection and transformation utilities."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMNS = {
    "claim_count",
    "claim_amount",
    "frequency_target",
    "severity_target",
    "pure_premium_target",
}


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split feature columns into numeric and categorical groups."""
    candidate_columns = [column for column in df.columns if column not in TARGET_COLUMNS and column != "policy_id"]
    numeric_features = [column for column in candidate_columns if pd.api.types.is_numeric_dtype(df[column])]
    categorical_features = [column for column in candidate_columns if column not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build a reusable sklearn preprocessor for mixed MTPL features."""
    numeric_features, categorical_features = get_feature_columns(df)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
