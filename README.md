# Motor Insurance Pricing Demo

A clean, production-style Python project for building a baseline motor third-party liability (MTPL) pricing model. The repository is organized around a strong data foundation first: raw source files, interim joins, curated policy and claim datasets, and model-ready tables are kept separate so training can start from reproducible inputs.

## Project Overview

In motor insurance, pricing often starts by decomposing expected claims into two components:

- `Frequency`: how often claims occur for a policy over the exposure period.
- `Severity`: the average cost of a claim when a claim happens.
- `Pure premium`: the expected loss cost per exposure unit, computed as `frequency × severity`.

This project follows that actuarial decomposition with two baseline models:

- A `PoissonRegressor` for claim frequency.
- A `GammaRegressor` for claim severity.

The output is a pure premium estimate built from a policy-level modeling table with aggregated claim severity.

## Architecture

The repository is split into modular layers that support a training-first workflow:

- `src/data`: raw data loading and preprocessing.
- `src/features`: feature grouping and sklearn preprocessing pipelines.
- `src/models`: baseline training, persistence, and batch prediction.
- `pipelines`: single-entry orchestration scripts for baseline training.
- `docker`: reproducible container packaging for local execution or lightweight cloud deployment.

## Data Foundation

The project keeps each data stage separate so policy-level and claim-level datasets do not get mixed:

- `data/raw`: immutable source files exactly as received.
- `data/interim`: mechanical joins and staging outputs that are still close to the source schema.
- `data/curated`: stable business-level datasets such as one row per policy and one row per claim.
- `data/model_input`: final model-ready datasets consumed by training.
- `data/simulated`: reserved for future artificial scenario generation.
- `data/scripts`: executable utilities split by purpose into analysis, wrangling, validation, and simulation.

This structure avoids duplicating `exposure` across repeated claim rows in the modeling dataset. Severity is aggregated back to policy level before training.

## Folder Structure

```text
data/
  raw/
  interim/
  curated/
  model_input/
  simulated/
  scripts/
    analysis/
      summary_statistics.py
    wrangling/
      join_data.py
      build_curated_datasets.py
    validation/
    simulation/
src/
  data/
    __init__.py
    load_data.py
    preprocess.py
  features/
    __init__.py
    feature_engineering.py
  models/
    __init__.py
    train.py
    predict.py
pipelines/
  __init__.py
  train_pipeline.py
config/
  config.yaml
notebooks/
  eda.ipynb
reports/
docker/
  Dockerfile
.dockerignore
README.md
requirements.txt
```

## Data Assumption

Place the source MTPL files in `data/raw/`:

- `freMTPL2freq.csv`: policy-level frequency and exposure data.
- `freMTPL2sev.csv`: claim-level severity data.

The training pipeline builds a policy-level modeling table with fields such as:

- `driver_age`
- `vehicle_age`
- `vehicle_power`
- `bonus_malus`
- `density`
- `exposure`
- `claim_count`
- `claim_amount`

If the raw files are not present, the training pipeline generates a synthetic MTPL-like policy-level dataset so the project can still run end-to-end.

## How To Run Locally

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the baseline training pipeline:

```bash
c:/Users/Ajeet/Desktop/FrenchMotor/.venv/Scripts/python.exe -m pipelines.train_pipeline
```

4. Build the curated policy-level and claim-level datasets explicitly:

```bash
c:/Users/Ajeet/Desktop/FrenchMotor/.venv/Scripts/python.exe data/scripts/wrangling/build_curated_datasets.py
```

5. Inspect summary statistics:

```bash
c:/Users/Ajeet/Desktop/FrenchMotor/.venv/Scripts/python.exe data/scripts/analysis/summary_statistics.py
```

Artifacts are written to config-driven locations under `data/interim/`, `data/curated/`, `data/model_input/`, and `reports/models/`.

## Docker Usage

Build the image:

```bash
docker build -f docker/Dockerfile -t mtpl-pricing-demo .
```

Run the default training pipeline:

```bash
docker run --rm mtpl-pricing-demo
```

## Cloud Readiness

The project is designed to be easy to run in containerized environments such as Google Cloud Run or other managed execution platforms:

- No hardcoded absolute paths.
- Config-driven directories via `config/config.yaml`.
- Single-entry pipeline commands for training and data preparation.
- Container-friendly stdout reporting for orchestration logs.

For a fuller cloud deployment, you could add:

- Object storage for raw and staged data.
- A model registry.
- CI/CD for container builds and training workflows.

## Notes For Extension

This starter implementation is intentionally simple and extendable. Natural next steps include:

- Swapping baseline GLMs with `XGBoost` models.
- Logging experiments with MLflow.
- Adding validation checks for policy-level and claim-level dataset consistency.
- Serving predictions with FastAPI.
- Adding unit tests and CI workflows.
