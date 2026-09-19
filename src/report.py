import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURES_PATH, METRICS_PATH, REPORTS_DIR, TARGET, TRAIN_PATH


def money(value: float) -> str:
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


def safe_corr(df: pd.DataFrame, target: str) -> pd.Series:
    numeric = df.select_dtypes(include=np.number)
    if target not in numeric:
        return pd.Series(dtype=float)
    return numeric.corr(numeric_only=True)[target].drop(target).sort_values(key=lambda s: s.abs(), ascending=False)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    result = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_rate": missing / len(df),
        }
    )
    return result[result["missing_count"] > 0].sort_values("missing_rate", ascending=False)


def neighborhood_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "Neighborhood" not in df:
        return pd.DataFrame()
    grouped = (
        df.groupby("Neighborhood")
        .agg(
            homes=(TARGET, "size"),
            median_price=(TARGET, "median"),
            mean_price=(TARGET, "mean"),
            median_living_area=("GrLivArea", "median"),
            median_quality=("OverallQual", "median"),
        )
        .sort_values("median_price", ascending=False)
    )
    grouped["median_price_per_sqft"] = grouped["median_price"] / grouped["median_living_area"]
    return grouped


def buyer_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    if "Neighborhood" not in df:
        return pd.DataFrame()

    summary = neighborhood_summary(df)
    market_median = df[TARGET].median()
    area_median = df["GrLivArea"].median() if "GrLivArea" in df else np.nan

    candidates = []
    for neighborhood, row in summary.iterrows():
        if row["homes"] < 10:
            continue
        price = row["median_price"]
        area = row["median_living_area"]
        quality = row["median_quality"]
        if price <= market_median and area >= area_median:
            segment = "Value buy"
        elif quality >= 7 and price >= market_median:
            segment = "Premium quality"
        elif price <= market_median * 0.9:
            segment = "Budget friendly"
        else:
            segment = "Balanced shortlist"
        candidates.append(
            {
                "segment": segment,
                "neighborhood": neighborhood,
                "median_price": price,
                "median_living_area": area,
                "median_quality": quality,
                "homes": row["homes"],
                "price_per_sqft": row["median_price_per_sqft"],
            }
        )
    result = pd.DataFrame(candidates)
    if result.empty:
        return result
    priority = {"Value buy": 0, "Budget friendly": 1, "Balanced shortlist": 2, "Premium quality": 3}
    result["priority"] = result["segment"].map(priority)
    return result.sort_values(["priority", "price_per_sqft", "median_price"]).drop(columns="priority").head(12)


def write_report(output_path: Path) -> Path:
    df = pd.read_csv(TRAIN_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    profile = json.loads(FEATURES_PATH.read_text(encoding="utf-8")) if FEATURES_PATH.exists() else {}

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    corr = safe_corr(df, TARGET)
    missing = missing_summary(df)
    neighborhoods = neighborhood_summary(df)
    recommendations = buyer_recommendations(df)

    lines = [
        "# House Price Data Analysis Report",
        "",
        "## Executive Summary",
        f"- Dataset used: `{TRAIN_PATH.name}` with {len(df):,} properties and {df.shape[1] - 1:,} explanatory columns.",
        f"- Target variable: `{TARGET}`, ranging from {money(df[TARGET].min())} to {money(df[TARGET].max())}; median price is {money(df[TARGET].median())}.",
        f"- Average sale price is {money(df[TARGET].mean())}, which is higher than the median, so the market is right-skewed by expensive homes.",
        f"- The strongest numeric price relationships are led by {', '.join(corr.head(5).index.tolist())}.",
        "",
        "## Data Quality",
        f"- Numeric columns: {len(numeric_cols):,}. Categorical columns: {len(categorical_cols):,}.",
        f"- Columns with missing values: {len(missing):,}. Most missing fields represent optional amenities such as pool, alley, fence, fireplace, garage, and basement attributes.",
        f"- The model pipeline handles missing values with median imputation for numeric fields and explicit missing categories for categorical fields.",
        "",
        "### Highest Missing Value Rates",
        "",
        "| Feature | Missing Count | Missing Rate |",
        "| --- | ---: | ---: |",
    ]

    for feature, row in missing.head(12).iterrows():
        lines.append(f"| {feature} | {int(row['missing_count']):,} | {pct(row['missing_rate'])} |")

    lines.extend(
        [
            "",
            "## Price Distribution",
            f"- 25th percentile: {money(df[TARGET].quantile(0.25))}.",
            f"- Median: {money(df[TARGET].quantile(0.50))}.",
            f"- 75th percentile: {money(df[TARGET].quantile(0.75))}.",
            f"- 90th percentile: {money(df[TARGET].quantile(0.90))}.",
            "",
            "## Feature Relationships",
            "",
            "| Feature | Correlation With Price | Interpretation |",
            "| --- | ---: | --- |",
        ]
    )

    interpretations = {
        "OverallQual": "Higher material and finish quality strongly increases expected price.",
        "GrLivArea": "Larger above-grade living area raises price substantially.",
        "GarageCars": "More garage capacity is associated with higher-value homes.",
        "GarageArea": "Larger garage area tracks with higher prices.",
        "TotalBsmtSF": "More basement area generally increases valuation.",
        "1stFlrSF": "Larger first-floor area is positively valued.",
        "YearBuilt": "Newer homes generally sell at higher prices.",
        "FullBath": "More full bathrooms support higher valuations.",
        "YearRemodAdd": "More recent remodels are associated with stronger sale prices.",
        "TotRmsAbvGrd": "More rooms above grade usually reflect larger and more valuable homes.",
    }
    for feature, value in corr.head(12).items():
        lines.append(f"| {feature} | {value:.3f} | {interpretations.get(feature, 'Meaningful numeric association with sale price.')} |")

    if not neighborhoods.empty:
        lines.extend(
            [
                "",
                "## Area Analysis",
                "",
                "| Neighborhood | Homes | Median Price | Median Living Area | Median Quality | Median Price Per Sq Ft |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for neighborhood, row in neighborhoods.head(15).iterrows():
            lines.append(
                f"| {neighborhood} | {int(row['homes']):,} | {money(row['median_price'])} | "
                f"{row['median_living_area']:,.0f} | {row['median_quality']:.1f} | {money(row['median_price_per_sqft'])} |"
            )

    if metrics:
        lines.extend(
            [
                "",
                "## Machine Learning Model",
                "- Algorithm: log-target ensemble regression combining histogram gradient boosting and random forest inside a scikit-learn pipeline.",
                "- Preprocessing: generated housing features, numeric imputation, robust scaling, categorical missing handling, and one-hot encoding.",
                f"- Validation RMSLE: {metrics.get('rmsle', 0):.4f}.",
                f"- Validation MAE: {money(metrics.get('mae', 0))}.",
                f"- Validation R2: {metrics.get('r2', 0):.3f}.",
                f"- Cross-validation RMSLE mean: {metrics.get('cv_rmsle_mean', 0):.4f}.",
            ]
        )

    lines.extend(
        [
            "",
            "## Buying Suggestions",
            "- Prioritize `OverallQual`, `GrLivArea`, `TotalBsmtSF`, garage capacity, and neighborhood before smaller cosmetic attributes because these show the strongest relationship with price.",
            "- For budget-sensitive buyers, target neighborhoods with below-median prices but acceptable living area and quality rather than simply choosing the cheapest property.",
            "- For family buyers, shortlist homes with at least median living area, adequate bedrooms and bathrooms, and garage capacity before optimizing for extras such as porch, deck, or pool.",
            "- For premium buyers, pay attention to high-quality neighborhoods and newer or recently remodeled homes; the data shows quality and age-related variables carry large pricing power.",
            "",
            "### Customer Shortlist By Area And Price",
            "",
            "| Segment | Neighborhood | Median Price | Median Living Area | Median Quality | Price Per Sq Ft |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for _, row in recommendations.iterrows():
        lines.append(
            f"| {row['segment']} | {row['neighborhood']} | {money(row['median_price'])} | "
            f"{row['median_living_area']:,.0f} | {row['median_quality']:.1f} | {money(row['price_per_sqft'])} |"
        )

    lines.extend(
        [
            "",
            "## Deliverables Generated",
            f"- Model artifact: `artifacts/house_price_model.joblib`.",
            f"- Model metrics: `artifacts/metrics.json`.",
            f"- Validation audit: `reports/validation_predictions.csv`.",
            f"- Feature profile for the app: `artifacts/feature_profile.json` with {len(profile.get('feature_columns', [])):,} model input columns.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate the data analysis and customer recommendation report.")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "data_analysis_report.md")
    args = parser.parse_args()
    path = write_report(args.output)
    print(f"Saved report to {path}")


if __name__ == "__main__":
    main()
