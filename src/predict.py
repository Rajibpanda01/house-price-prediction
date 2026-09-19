import argparse
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import pandas as pd

from src.config import ID_COLUMN, MODEL_PATH, SUBMISSIONS_DIR, TEST_PATH
from src.model import predict_prices


def make_predictions(input_path: Path, output_path: Path, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    model = joblib.load(model_path)
    data = pd.read_csv(input_path)
    predictions = predict_prices(model, data)
    submission = pd.DataFrame({ID_COLUMN: data[ID_COLUMN], "SalePrice": predictions})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def main():
    parser = argparse.ArgumentParser(description="Generate price predictions for a CSV file.")
    parser.add_argument("--input", type=Path, default=TEST_PATH)
    parser.add_argument("--output", type=Path, default=SUBMISSIONS_DIR / "advanced_submission.csv")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    result = make_predictions(args.input, args.output, args.model_path)
    print(f"Saved {len(result):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
