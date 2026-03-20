"""Build curated policy-level and claim-level datasets from the raw MTPL sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def main(profile: str | None = None) -> None:
    """Create curated datasets with clear entity-level granularity."""
    config = load_project_config(PROJECT_ROOT, profile=profile)
    freq_df, sev_df, _, _ = load_raw_datasets(config)

    claim_level_df = standardize_source_columns(build_claim_level_dataset(freq_df, sev_df))
    policy_level_df = standardize_source_columns(build_policy_level_dataset(freq_df, sev_df))

    curated_dir = PROJECT_ROOT / config["paths"]["curated_data_dir"]
    curated_dir.mkdir(parents=True, exist_ok=True)

    claim_level_path = curated_dir / "mtpl_claim_level_curated.csv"
    policy_level_path = curated_dir / "mtpl_policy_level_curated.csv"

    claim_level_df.to_csv(claim_level_path, index=False)
    policy_level_df.to_csv(policy_level_path, index=False)

    print(f"Saved curated claim-level dataset to {claim_level_path}")
    print(f"  Shape: {claim_level_df.shape}")
    print(f"Saved curated policy-level dataset to {policy_level_path}")
    print(f"  Shape: {policy_level_df.shape}")
    print("\nPolicy-level dataset keeps one row per policy and aggregated claim_amount.")
    print("Claim-level dataset keeps one row per claim with policy attributes attached.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build curated MTPL datasets.")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional profile under config/profiles (for example: dev, strict).",
    )
    args = parser.parse_args()
    main(profile=args.profile)
