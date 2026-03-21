#!/usr/bin/env python3
"""Hyperparameter tuning for XGBoost frequency model (subsampled).

Saves best model to reports/models/xgb_freq_tuned.pkl and params to reports/models/xgb_freq_tuning.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from scipy.stats import randint, uniform
import pickle
from xgboost import XGBRegressor

from src.features.feature_engineering import build_preprocessor

ROOT = Path('.')
OUT = Path('reports/models')
OUT.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = Path('config/config.yaml')

# Load data
df = pd.read_csv(Path('data/model_input/mtpl_model_input.csv'))
# target
y = df['claim_count']
weights = df['exposure']

# subsample for tuning for speed
max_rows = 100000
if len(df) > max_rows:
    df_sub = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
else:
    df_sub = df.copy()

X_sub = df_sub
y_sub = X_sub['claim_count']
w_sub = X_sub['exposure']

# preprocessor
pre = build_preprocessor(X_sub)

est = Pipeline(steps=[('preprocessor', pre), ('model', XGBRegressor(objective='count:poisson', n_jobs=1, verbosity=0))])

param_dist = {
    'model__n_estimators': randint(100, 301),
    'model__max_depth': randint(3, 9),
    'model__learning_rate': uniform(0.01, 0.19),
    'model__subsample': uniform(0.6, 0.4),
    'model__colsample_bytree': uniform(0.6, 0.4),
    'model__reg_lambda': uniform(0.0, 2.0),
}

rs = RandomizedSearchCV(
    est,
    param_distributions=param_dist,
    n_iter=12,
    cv=3,
    scoring='neg_mean_absolute_error',
    verbose=2,
    random_state=42,
    n_jobs=1,
)

print('Starting RandomizedSearchCV (subsampled) ...')
rs.fit(X_sub, y_sub, model__sample_weight=w_sub)

print('Best score:', rs.best_score_)
print('Best params:', rs.best_params_)

# Save results
with open(OUT / 'xgb_freq_tuning.json', 'w') as f:
    json.dump({'best_score': float(rs.best_score_), 'best_params': rs.best_params_}, f, indent=2)

with open(OUT / 'xgb_freq_tuned.pkl', 'wb') as f:
    pickle.dump({'best_estimator': rs.best_estimator_, 'cv_results': rs.cv_results_}, f)

print('Tuning complete. Artifacts saved to', OUT)
