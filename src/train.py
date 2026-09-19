import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "reports" / ".matplotlib"))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from src.config import ARTIFACT_DIR, FEATURES_PATH, METRICS_PATH, MODEL_PATH, RANDOM_STATE, REPORTS_DIR
from src.data import load_training_data, save_feature_profile, split_features_target
from src.model import build_model


def regression_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    rmsle = float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(np.maximum(y_pred, 0)))))
    return {"rmse": rmse, "mae": mae, "r2": r2, "rmsle": rmsle}


def train(output_model: Path = MODEL_PATH, metrics_path: Path = METRICS_PATH) -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_training_data()
    save_feature_profile(df, FEATURES_PATH)
    x, y = split_features_target(df)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.18,
        random_state=RANDOM_STATE,
    )

    model = build_model()
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = -cross_val_score(
        model,
        x_train,
        y_train,
        scoring="neg_root_mean_squared_log_error",
        cv=cv,
        n_jobs=1,
    )

    model.fit(x_train, y_train)
    valid_predictions = model.predict(x_valid)
    metrics = regression_metrics(y_valid, valid_predictions)
    metrics["cv_rmsle_mean"] = float(cv_scores.mean())
    metrics["cv_rmsle_std"] = float(cv_scores.std())
    metrics["train_rows"] = int(len(x_train))
    metrics["validation_rows"] = int(len(x_valid))

    final_model = build_model()
    final_model.fit(x, y)
    joblib.dump(final_model, output_model)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = pd.DataFrame(
        {
            "actual_price": y_valid,
            "predicted_price": valid_predictions,
            "absolute_error": np.abs(y_valid - valid_predictions),
        }
    ).sort_values("absolute_error", ascending=False)
    report.to_csv(REPORTS_DIR / "validation_predictions.csv", index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the house price prediction model.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    args = parser.parse_args()

    metrics = train(args.model_path, args.metrics_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
