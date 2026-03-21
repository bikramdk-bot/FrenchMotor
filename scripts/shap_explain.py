#!/usr/bin/env python3
"""Compute SHAP explanations for XGBoost frequency model and save summary plots/CSV.

Requires `shap` package.
"""
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

PKL = Path('reports/models/baseline_model.pkl')
TUNED = Path('reports/models/xgb_freq_tuned.pkl')
OUT = Path('reports/models')
OUT.mkdir(parents=True, exist_ok=True)

# load model (prefer tuned if exists)
if TUNED.exists():
    with open(TUNED, 'rb') as f:
        tuned = pickle.load(f)
    pipeline = tuned.get('best_estimator', tuned)
else:
    with open(PKL, 'rb') as f:
        payload = pickle.load(f)
    pipeline = payload['frequency_model']

# load some data sample
df = pd.read_csv('data/model_input/mtpl_model_input.csv')
sample = df.sample(n=min(2000, len(df)), random_state=42).reset_index(drop=True)

pre = pipeline.named_steps['preprocessor']
model = pipeline.named_steps['model']

# get transformed features and names
X_trans = pre.transform(sample)
try:
    feature_names = pre.get_feature_names_out(sample.columns)
except Exception:
    # fallback: use original columns
    feature_names = list(sample.columns)

# SHAP TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_trans)

# summary plot
plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, features=X_trans, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(OUT / 'shap_summary.png', dpi=150)
plt.close()

# bar plot of mean abs shap
plt.figure(figsize=(6,6))
shap.summary_plot(shap_values, features=X_trans, feature_names=feature_names, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig(OUT / 'shap_bar.png', dpi=150)
plt.close()

# save mean abs shap values to CSV
abs_mean = np.mean(np.abs(shap_values), axis=0)
if hasattr(feature_names, '__len__') and len(abs_mean) == len(feature_names):
    import pandas as pd
    df_shap = pd.Series(abs_mean, index=feature_names).sort_values(ascending=False)
    df_shap.to_csv(OUT / 'shap_mean_abs.csv')

print('SHAP plots and CSV saved to', OUT)
