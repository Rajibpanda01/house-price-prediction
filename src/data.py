import json
from pathlib import Path

import pandas as pd

from src.config import ID_COLUMN, TARGET, TRAIN_PATH


def load_training_data(path: Path = TRAIN_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    return x, y


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in {TARGET}]


def save_feature_profile(df: pd.DataFrame, path: Path) -> dict:
    features = model_feature_columns(df)
    profile = {
        "feature_columns": features,
        "numeric_features": df[features].select_dtypes(exclude="object").columns.tolist(),
        "categorical_features": df[features].select_dtypes(include="object").columns.tolist(),
        "defaults": {},
        "options": {},
    }

    for column in features:
        series = df[column]
        if column == ID_COLUMN:
            profile["defaults"][column] = int(series.max()) + 1
        elif series.dtype == "object":
            mode = series.dropna().mode()
            profile["defaults"][column] = str(mode.iloc[0]) if not mode.empty else "NA"
            values = sorted(series.fillna("NA").astype(str).unique().tolist())
            profile["options"][column] = values
        else:
            median = series.median()
            profile["defaults"][column] = float(median) if pd.notna(median) else 0.0

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile
