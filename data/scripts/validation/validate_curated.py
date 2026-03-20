"""Validate curated MTPL datasets using rules from config/data_validation.yaml and optional profiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_project_config


def _null_fraction(df: pd.DataFrame, column: str) -> float:
    return float(df[column].isna().mean())


def validate_policy_level(policy_df: pd.DataFrame, rules: dict) -> list[str]:
    errors: list[str] = []

    key_col = rules["unique_key"]
    duplicated_keys = int(policy_df[key_col].duplicated().sum())
    if duplicated_keys > 0:
        errors.append(f"Policy-level key '{key_col}' has {duplicated_keys} duplicate rows.")

    for col in rules["required_non_null_columns"]:
        if col not in policy_df.columns:
            errors.append(f"Policy-level missing required column: {col}")
            continue
        if _null_fraction(policy_df, col) > rules["max_null_fraction"]:
            errors.append(
                f"Policy-level column '{col}' exceeds null fraction limit "
                f"({policy_df[col].isna().mean():.4f} > {rules['max_null_fraction']:.4f})."
            )

    if "exposure" in policy_df.columns:
        min_exposure = float(policy_df["exposure"].min())
        if min_exposure < float(rules["min_exposure"]):
            errors.append(
                f"Policy-level exposure min is {min_exposure:.6f}, below required {rules['min_exposure']}."
            )

    return errors


def validate_claim_level(claim_df: pd.DataFrame, rules: dict) -> list[str]:
    errors: list[str] = []

    for col in rules["required_non_null_columns"]:
        if col not in claim_df.columns:
            errors.append(f"Claim-level missing required column: {col}")
            continue
        if _null_fraction(claim_df, col) > rules["max_null_fraction"]:
            errors.append(
                f"Claim-level column '{col}' exceeds null fraction limit "
                f"({claim_df[col].isna().mean():.4f} > {rules['max_null_fraction']:.4f})."
            )

    if "claim_amount" in claim_df.columns:
        min_claim = float(claim_df["claim_amount"].min())
        if min_claim < float(rules["min_claim_amount"]):
            errors.append(
                f"Claim-level claim_amount min is {min_claim:.6f}, below required {rules['min_claim_amount']}."
            )

    return errors


def main(profile: str | None = None) -> None:
    """Validate curated datasets and print a concise quality report."""
    config = load_project_config(PROJECT_ROOT, profile=profile)
    curated_dir = PROJECT_ROOT / config["paths"]["curated_data_dir"]

    policy_path = curated_dir / "mtpl_policy_level_curated.csv"
    claim_path = curated_dir / "mtpl_claim_level_curated.csv"

    if not policy_path.exists() or not claim_path.exists():
        raise FileNotFoundError(
            "Curated datasets not found. Run data/scripts/wrangling/build_curated_datasets.py first."
        )

    policy_df = pd.read_csv(policy_path)
    claim_df = pd.read_csv(claim_path)

    policy_rules = config["validation"]["policy_level"]
    claim_rules = config["validation"]["claim_level"]

    errors = []
    errors.extend(validate_policy_level(policy_df, policy_rules))
    errors.extend(validate_claim_level(claim_df, claim_rules))

    profile_name = config.get("runtime", {}).get("profile_name", "base")
    print(f"Validation profile: {profile_name}")
    print(f"Policy-level rows: {len(policy_df)}")
    print(f"Claim-level rows: {len(claim_df)}")

    if errors:
        print("\nValidation status: FAILED")
        for idx, msg in enumerate(errors, start=1):
            print(f"  {idx}. {msg}")
        raise SystemExit(1)

    print("\nValidation status: PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate curated MTPL datasets.")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional profile under config/profiles (for example: dev, strict).",
    )
    args = parser.parse_args()
    main(profile=args.profile)
