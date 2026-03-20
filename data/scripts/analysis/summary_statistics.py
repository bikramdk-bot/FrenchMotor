"""Summary statistics for the MTPL raw, interim, and curated datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import (
    build_claim_level_dataset,
    build_policy_level_dataset,
    load_project_config,
    load_raw_datasets,
)
from src.data.preprocess import standardize_source_columns


def print_policy_frequency_summary(policy_df: pd.DataFrame) -> None:
    """Print policy-level claim count distribution."""
    print("Policy-Level Claim Distribution:")
    print(f"  Total unique policies: {len(policy_df)}")
    print(f"  Policies with no claims: {(policy_df['claim_count'] == 0).sum()}")
    print(f"  Policies with 1+ claims: {(policy_df['claim_count'] > 0).sum()}")
    print("\n  Claim count distribution:")

    max_claim_count = int(policy_df["claim_count"].max())
    for claim_count in range(min(6, max_claim_count + 1)):
        count = int((policy_df["claim_count"] == claim_count).sum())
        pct = 100 * count / len(policy_df)
        print(f"    {claim_count} claims: {count:,} policies ({pct:.2f}%)")


def print_claim_record_summary(claim_df: pd.DataFrame) -> None:
    """Print claim-level summary statistics."""
    print("\nClaim-Level Severity Summary:")
    print(f"  Total claim records: {len(claim_df)}")

    claims_per_policy = claim_df["policy_id"].value_counts()
    print(f"  Policies represented in severity data: {len(claims_per_policy)}")
    print(f"  Policies with multiple claims: {(claims_per_policy > 1).sum()}")
    print(f"  Max claims per policy: {claims_per_policy.max()}")
    print(f"  Avg claims per policy (when claims > 0): {claims_per_policy.mean():.2f}")

    print("\n  Top 10 policies by claim count:")
    for idx, (policy_id, count) in enumerate(claims_per_policy.head(10).items(), 1):
        print(f"    {idx}. Policy {policy_id}: {count} claims")

    claim_amounts = claim_df["claim_amount"].dropna()
    print("\nClaim Amount Statistics:")
    print(f"  Total claims: {len(claim_amounts)}")
    print(f"  Min claim: EUR {claim_amounts.min():.2f}")
    print(f"  Max claim: EUR {claim_amounts.max():.2f}")
    print(f"  Mean claim: EUR {claim_amounts.mean():.2f}")
    print(f"  Median claim: EUR {claim_amounts.median():.2f}")
    print(f"  Std dev: EUR {claim_amounts.std():.2f}")


def print_dataset_foundation_summary(policy_df: pd.DataFrame, claim_df: pd.DataFrame) -> None:
    """Print dataset grain and storage guidance summaries."""
    print("Dataset Foundation Summary:")
    print(f"  Curated policy-level rows: {len(policy_df)}")
    print(f"  Curated claim-level rows: {len(claim_df)}")
    print(f"  Policies with aggregated claim_amount > 0: {(policy_df['claim_amount'] > 0).sum()}")
    print(f"  Policies with claim_count > 0 but zero claim_amount: {((policy_df['claim_count'] > 0) & (policy_df['claim_amount'] == 0)).sum()}")


def main(profile: str | None = None) -> None:
    """Load the raw datasets and print summary statistics on the professional data layers."""
    config = load_project_config(PROJECT_ROOT, profile=profile)
    freq_df, sev_df, _, _ = load_raw_datasets(config)
    claim_df = standardize_source_columns(build_claim_level_dataset(freq_df, sev_df))
    policy_df = standardize_source_columns(build_policy_level_dataset(freq_df, sev_df))

    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print_dataset_foundation_summary(policy_df, claim_df)
    print()
    print_policy_frequency_summary(policy_df)
    print_claim_record_summary(claim_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print summary statistics for MTPL datasets.")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional profile under config/profiles (for example: dev, strict).",
    )
    args = parser.parse_args()
    main(profile=args.profile)
