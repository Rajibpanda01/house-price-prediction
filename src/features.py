import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds domain features while preserving original columns for the model."""

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        df = x.copy()

        def numeric(column: str, default: float = 0.0) -> pd.Series:
            if column in df:
                return pd.to_numeric(df[column], errors="coerce").fillna(default)
            return pd.Series(default, index=df.index)

        df["TotalSF"] = (
            numeric("TotalBsmtSF")
            + numeric("1stFlrSF")
            + numeric("2ndFlrSF")
        )
        df["TotalBathrooms"] = (
            numeric("FullBath")
            + 0.5 * numeric("HalfBath")
            + numeric("BsmtFullBath")
            + 0.5 * numeric("BsmtHalfBath")
        )
        df["HouseAgeAtSale"] = numeric("YrSold") - numeric("YearBuilt")
        df["YearsSinceRemodel"] = numeric("YrSold") - numeric("YearRemodAdd")
        df["HasGarage"] = (numeric("GarageArea") > 0).astype(int)
        df["HasBasement"] = (numeric("TotalBsmtSF") > 0).astype(int)
        df["HasFireplace"] = (numeric("Fireplaces") > 0).astype(int)
        df["HasPool"] = (numeric("PoolArea") > 0).astype(int)
        df["OverallScore"] = numeric("OverallQual") * numeric("OverallCond")
        df["LogLotArea"] = np.log1p(numeric("LotArea"))

        age_columns = ["HouseAgeAtSale", "YearsSinceRemodel"]
        for column in age_columns:
            df[column] = df[column].clip(lower=0)

        return df
