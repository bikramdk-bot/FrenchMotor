# Data Foundation Guide

This folder is the source of truth for all data work in the project.

Current phase:
- Focus only on data ingestion, wrangling, quality checks, and documentation.
- Do not run model training from this folder.
- Keep simulation outputs separate from real observed data.

## Folder Purpose

- raw/: Immutable source files as received.
  - freMTPL2freq.csv (policy-level frequency and exposure)
  - freMTPL2sev.csv (claim-level severity)
- interim/: Mechanical staging outputs close to raw schema.
  - mtpl_claim_level_joined.csv
- curated/: Business-ready datasets with clear grain.
  - mtpl_policy_level_curated.csv (one row per policy)
  - mtpl_claim_level_curated.csv (one row per claim)
- model_input/: Final model-ready tables derived from curated data.
- simulated/: Artificial or scenario-generated datasets.
- scripts/: Executable data utilities by purpose.
  - wrangling/: joins, reshaping, curated dataset builders
  - analysis/: profiling and summary statistics
  - validation/: data quality and key checks
  - simulation/: synthetic/scenario generation utilities

## Data Grain Contract

Keep entity grain explicit in file names and logic:
- policy_level: one row per policy_id
- claim_level: one row per claim record

Important:
- Exposure belongs to policy-level records.
- Do not divide exposure by claim count.
- For policy-level modeling tables, aggregate claim_amount by policy_id before joining.

## Script Entry Points

Run scripts from the repository root using the project virtual environment.

1) Build interim claim-level join:
- data/scripts/wrangling/join_data.py

2) Build curated datasets:
- data/scripts/wrangling/build_curated_datasets.py

3) Review summary statistics:
- data/scripts/analysis/summary_statistics.py

4) Validate curated datasets:
- data/scripts/validation/validate_curated.py

## Config Profiles

Scripts can consume layered configuration from:
- config/config.yaml (base defaults)
- config/data_validation.yaml (validation rules)
- config/profiles/<profile>.yaml (runtime overrides)

Use profile overrides when running scripts:
- --profile dev
- --profile strict

Example commands:
- c:/Users/Ajeet/Desktop/FrenchMotor/.venv/Scripts/python.exe data/scripts/wrangling/build_curated_datasets.py --profile dev
- c:/Users/Ajeet/Desktop/FrenchMotor/.venv/Scripts/python.exe data/scripts/analysis/summary_statistics.py --profile strict
- c:/Users/Ajeet/Desktop/FrenchMotor/.venv/Scripts/python.exe data/scripts/validation/validate_curated.py --profile strict

## Naming Rules

- Use lower_snake_case.
- Include grain in dataset names: policy_level or claim_level.
- Keep deterministic file names for repeatability.
- Avoid ad-hoc one-off filenames in committed outputs.

## Quality Checks (to add next)

Validation scripts should check:
- policy_id uniqueness in policy_level tables
- non-null key fields
- row count consistency between stages
- claim_count and claim_amount consistency rules
- duplicate detection and reporting

## Change Management

When adding or changing data logic:
- update this README first
- add or update a script under data/scripts/
- write outputs only to the correct stage folder
- keep raw files untouched

## Suggested Next Steps

1. Add validation scripts under data/scripts/validation/:
   - validate_policy_keys.py
   - validate_claim_consistency.py
   - validate_stage_rowcounts.py
2. Add one data dictionary CSV for curated datasets.
3. Freeze a versioned snapshot process for curated outputs.
4. Add simulation scripts only after validation checks are stable.
