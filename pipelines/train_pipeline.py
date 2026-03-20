"""Training pipeline entry point."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.load_data import load_config, load_raw_data
from src.data.preprocess import preprocess_data, save_model_input_data
from src.models.train import save_artifacts, train_models


def main() -> None:
    """Run the baseline training workflow."""
    config_path = Path("config/config.yaml")
    config = load_config(config_path)

    raw_df, raw_path = load_raw_data(config)
    model_input_df = preprocess_data(raw_df)
    model_input_path = save_model_input_data(model_input_df, config)

    artifacts = train_models(model_input_df, config)
    model_path = save_artifacts(artifacts, config)

    summary = {
        "raw_data_path": str(raw_path),
        "model_input_path": str(model_input_path),
        "model_path": str(model_path),
        "train_rows": artifacts["train_rows"],
        "test_rows": artifacts["test_rows"],
        "metrics": artifacts["metrics"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
