"""Build and inspect the interim claim-level join between MTPL frequency and severity data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import build_claim_level_dataset, load_project_config, load_raw_datasets
from src.data.preprocess import standardize_source_columns


def main(profile: str | None = None) -> None:
    """Create the interim claim-level joined dataset and display a compact preview."""
    config = load_project_config(PROJECT_ROOT, profile=profile)
    freq_df, sev_df, frequency_path, severity_path = load_raw_datasets(config)

    joined_df = standardize_source_columns(build_claim_level_dataset(freq_df, sev_df))

    interim_dir = PROJECT_ROOT / config["paths"]["interim_data_dir"]
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_path = interim_dir / "mtpl_claim_level_joined.csv"
    joined_df.to_csv(output_path, index=False)

    print(f"Loaded frequency data from {frequency_path}")
    print(f"Loaded severity data from {severity_path}")
    print(f"Saved interim claim-level join to {output_path}")
    print(f"Joined shape: {joined_df.shape}")
    print(f"Columns: {joined_df.columns.tolist()}\n")

    print("Sample of joined data (first 5 rows):")
    print(joined_df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build interim claim-level joined MTPL dataset.")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional profile under config/profiles (for example: dev, strict).",
    )
    args = parser.parse_args()
    main(profile=args.profile)
