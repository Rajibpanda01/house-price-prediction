import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds domain features while preserving original columns for the model."""

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        df = x.copy()

        df["TotalSF"] = (
            df.get("TotalBsmtSF", 0).fillna(0)
            + df.get("1stFlrSF", 0).fillna(0)
            + df.get("2ndFlrSF", 0).fillna(0)
        )
        df["TotalBathrooms"] = (
            df.get("FullBath", 0).fillna(0)
            + 0.5 * df.get("HalfBath", 0).fillna(0)
            + df.get("BsmtFullBath", 0).fillna(0)
            + 0.5 * df.get("BsmtHalfBath", 0).fillna(0)
        )
        df["HouseAgeAtSale"] = df.get("YrSold", 0).fillna(0) - df.get("YearBuilt", 0).fillna(0)
        df["YearsSinceRemodel"] = df.get("YrSold", 0).fillna(0) - df.get("YearRemodAdd", 0).fillna(0)
        df["HasGarage"] = (df.get("GarageArea", 0).fillna(0) > 0).astype(int)
        df["HasBasement"] = (df.get("TotalBsmtSF", 0).fillna(0) > 0).astype(int)
        df["HasFireplace"] = (df.get("Fireplaces", 0).fillna(0) > 0).astype(int)
        df["HasPool"] = (df.get("PoolArea", 0).fillna(0) > 0).astype(int)
        df["OverallScore"] = df.get("OverallQual", 0).fillna(0) * df.get("OverallCond", 0).fillna(0)
        df["LogLotArea"] = np.log1p(df.get("LotArea", 0).fillna(0))

        age_columns = ["HouseAgeAtSale", "YearsSinceRemodel"]
        for column in age_columns:
            df[column] = df[column].clip(lower=0)

        return df
