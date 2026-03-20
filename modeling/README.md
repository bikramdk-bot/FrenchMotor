# Modeling Workspace

This folder is the dedicated home for all model development work.

Scope now:
- baseline model training design
- feature and target selection experiments
- evaluation and model comparison
- inference utilities

Out of scope for now:
- deployment orchestration
- retraining automation
- production monitoring

## Structure

- training/: training scripts and experiment runners
- evaluation/: metrics, diagnostics, and comparison scripts
- inference/: prediction utilities and batch scoring helpers
- configs/: modeling-specific YAML config files
- artifacts/: local model artifacts from experimental runs (gitignored recommended)

## Data Contract

Modeling should read only from:
- data/model_input/
- optionally data/curated/ for exploratory checks

Modeling should never write into:
- data/raw/
- data/interim/
- data/curated/

## Suggested Next Step

Create one baseline training entry script in training/ that:
1. reads data/model_input/mtpl_model_input.csv
2. trains frequency and severity baselines
3. writes artifacts under modeling/artifacts/
4. saves evaluation summary under modeling/evaluation/
