#!/usr/bin/env python3
"""Inspect trained model pickle, export metrics and diagnostics.

Save this as `scripts/inspect_model.py` and run from repository root.
"""
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PKL = Path("reports/models/baseline_model.pkl")
MODEL_INPUT = Path("data/model_input/mtpl_model_input.csv")
OUT_DIR = Path("reports/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# load pickle (only trust files you created)
with open(PKL, "rb") as f:
    payload = pickle.load(f)

# top-level summary
print("Top-level keys:", list(payload.keys()))
metrics = payload.get("metrics", {})
print("Metrics:")
print(json.dumps(metrics, indent=2, default=str))

# save metrics to JSON
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)

# load model input (to score and plot)
df = pd.read_csv(MODEL_INPUT)

freq_model = payload.get("frequency_model")
sev_model = payload.get("severity_model")

# Predict
pred_claim_count = np.clip(freq_model.predict(df), a_min=0.0, a_max=None)
pred_freq = pred_claim_count / df["exposure"].clip(lower=1e-6)
if sev_model is not None:
    pred_sev = np.clip(sev_model.predict(df), a_min=0.0, a_max=None)
else:
    pred_sev = np.zeros(len(df))

pred_pp = pred_freq * pred_sev

# Quick aggregate check on test/train sizes if present
print("train_rows:", payload.get("train_rows"))
print("test_rows:", payload.get("test_rows"))

# Basic diagnostics on a sample subset (to keep plots readable)
sample = df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)
sf = sample.copy()
pred_counts_sample = np.clip(freq_model.predict(sf), a_min=0.0, a_max=None)
pred_freq_sample = pred_counts_sample / sf["exposure"].clip(lower=1e-6)
sf["pred_freq"] = pd.Series(pred_freq_sample).clip(lower=0.0)
sf["pred_sev"] = (pd.Series(np.clip(sev_model.predict(sf), a_min=0.0, a_max=None)) if sev_model is not None else pd.Series(np.zeros(len(sf))))
sf["pred_pp"] = sf["pred_freq"] * sf["pred_sev"]

# Plot: predicted vs actual pure premium
plt.figure(figsize=(6, 6))
plt.scatter(sf["pure_premium_target"], sf["pred_pp"], alpha=0.4, s=8)
mx = max(sf["pure_premium_target"].max(), sf["pred_pp"].max())
plt.plot([0, mx], [0, mx], color="k", linewidth=0.8)
plt.xlabel("Observed pure premium")
plt.ylabel("Predicted pure premium")
plt.title("Predicted vs Observed Pure Premium (sample)")
plt.tight_layout()
plt.savefig(OUT_DIR / "pred_vs_actual_pure_premium.png", dpi=150)
plt.close()

# Residuals histogram
resid = sf["pure_premium_target"] - sf["pred_pp"]
plt.figure(figsize=(6, 4))
plt.hist(resid.clip(lower=-1e6, upper=1e6), bins=80)
plt.title("Pure premium residuals (sample)")
plt.tight_layout()
plt.savefig(OUT_DIR / "residuals_pure_premium.png", dpi=150)
plt.close()

# Feature importance (if available)
def extract_feature_importances(pipeline):
    try:
        pre = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
    except Exception:
        return None, None
    # feature names (best effort)
    try:
        feat_names = pre.get_feature_names_out(df.columns)
    except Exception:
        try:
            feat_names = df.columns
        except Exception:
            feat_names = None
    try:
        fi = model.feature_importances_
        if feat_names is not None and len(fi) == len(feat_names):
            fi_series = pd.Series(fi, index=feat_names).sort_values(ascending=False)
        else:
            fi_series = pd.Series(fi).sort_values(ascending=False)
        return fi_series, model
    except Exception:
        return None, None

fi_freq, mod_freq = extract_feature_importances(freq_model)
if fi_freq is not None:
    topk = fi_freq.head(30)
    plt.figure(figsize=(6, max(4, 0.2 * len(topk))))
    topk.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Frequency model feature importances (top 30)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "freq_feature_importances.png", dpi=150)
    plt.close()
    topk.to_csv(OUT_DIR / "freq_feature_importances.csv")

fi_sev, mod_sev = (extract_feature_importances(sev_model) if sev_model is not None else (None, None))
if fi_sev is not None:
    topk = fi_sev.head(30)
    plt.figure(figsize=(6, max(4, 0.2 * len(topk))))
    topk.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Severity model feature importances (top 30)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sev_feature_importances.png", dpi=150)
    plt.close()
    topk.to_csv(OUT_DIR / "sev_feature_importances.csv")

print("Diagnostics saved to:", OUT_DIR)
