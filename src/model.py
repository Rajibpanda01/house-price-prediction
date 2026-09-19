import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config import RANDOM_STATE
from src.features import HouseFeatureEngineer


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, make_column_selector(dtype_exclude=object)),
            ("cat", categorical_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model() -> TransformedTargetRegressor:
    base_pipeline = Pipeline(
        steps=[
            ("features", HouseFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            (
                "regressor",
                VotingRegressor(
                    estimators=[
                        (
                            "hgb",
                            HistGradientBoostingRegressor(
                                learning_rate=0.045,
                                max_iter=360,
                                l2_regularization=0.05,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                        (
                            "rf",
                            RandomForestRegressor(
                                n_estimators=220,
                                min_samples_leaf=2,
                                max_features="sqrt",
                                n_jobs=1,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ],
                    n_jobs=1,
                ),
            ),
        ]
    )

    return TransformedTargetRegressor(
        regressor=base_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def predict_prices(model, records: pd.DataFrame) -> np.ndarray:
    predictions = model.predict(records)
    return np.maximum(predictions, 0)
