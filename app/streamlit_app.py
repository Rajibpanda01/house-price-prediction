import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd
import streamlit as st

from src.config import FEATURES_PATH, METRICS_PATH, MODEL_PATH, TARGET, TRAIN_PATH
from src.model import predict_prices


st.set_page_config(page_title="House Price Intelligence", page_icon="H", layout="wide")


@st.cache_data
def load_profile():
    return json.loads(Path(FEATURES_PATH).read_text(encoding="utf-8"))


@st.cache_data
def load_metrics():
    if not Path(METRICS_PATH).exists():
        return {}
    return json.loads(Path(METRICS_PATH).read_text(encoding="utf-8"))


@st.cache_data
def load_training_preview():
    return pd.read_csv(TRAIN_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def money(value):
    return f"${value:,.0f}"


def ensure_artifacts():
    missing = [path for path in [MODEL_PATH, FEATURES_PATH] if not Path(path).exists()]
    if missing:
        st.error("Model artifacts are missing. Run `python3 -m src.train` before opening the app.")
        st.stop()


ensure_artifacts()
profile = load_profile()
metrics = load_metrics()
train_df = load_training_preview()
model = load_model()

st.title("House Price Intelligence")
st.caption("Client-ready valuation workspace powered by an end-to-end scikit-learn pipeline.")

metric_cols = st.columns(4)
metric_cols[0].metric("Validation RMSLE", f"{metrics.get('rmsle', 0):.4f}")
metric_cols[1].metric("Validation MAE", money(metrics.get("mae", 0)))
metric_cols[2].metric("Validation R2", f"{metrics.get('r2', 0):.3f}")
metric_cols[3].metric("Training Records", f"{len(train_df):,}")

tab_predict, tab_batch, tab_insights = st.tabs(["Single Property", "Batch Scoring", "Market Insights"])

with tab_predict:
    left, right = st.columns([0.38, 0.62])
    defaults = profile["defaults"].copy()
    options = profile["options"]

    with left:
        st.subheader("Property Details")
        defaults["OverallQual"] = st.slider("Overall quality", 1, 10, int(defaults.get("OverallQual", 6)))
        defaults["OverallCond"] = st.slider("Overall condition", 1, 10, int(defaults.get("OverallCond", 5)))
        defaults["GrLivArea"] = st.number_input("Living area (sq ft)", 300, 6000, int(defaults.get("GrLivArea", 1500)), 50)
        defaults["TotalBsmtSF"] = st.number_input("Basement area (sq ft)", 0, 4000, int(defaults.get("TotalBsmtSF", 900)), 50)
        defaults["GarageCars"] = st.slider("Garage capacity", 0, 5, int(defaults.get("GarageCars", 2)))
        defaults["YearBuilt"] = st.number_input("Year built", 1870, 2026, int(defaults.get("YearBuilt", 1973)))
        defaults["YearRemodAdd"] = st.number_input("Remodel year", 1870, 2026, int(defaults.get("YearRemodAdd", 1994)))
        defaults["Neighborhood"] = st.selectbox(
            "Neighborhood",
            options.get("Neighborhood", ["NA"]),
            index=options.get("Neighborhood", ["NA"]).index(defaults.get("Neighborhood", options.get("Neighborhood", ["NA"])[0])),
        )
        defaults["KitchenQual"] = st.selectbox("Kitchen quality", options.get("KitchenQual", ["TA"]))
        defaults["ExterQual"] = st.selectbox("Exterior quality", options.get("ExterQual", ["TA"]))

    record = pd.DataFrame([{column: defaults.get(column) for column in profile["feature_columns"]}])
    estimate = predict_prices(model, record)[0]

    with right:
        st.subheader("Estimated Valuation")
        st.markdown(f"<h1 style='margin-bottom:0'>{money(estimate)}</h1>", unsafe_allow_html=True)
        st.caption("Estimate uses the same preprocessing and feature engineering as model training.")

        comparison = train_df[[TARGET, "Neighborhood", "OverallQual", "GrLivArea"]].copy()
        comparison["distance"] = (
            (comparison["OverallQual"] - defaults["OverallQual"]).abs() * 12000
            + (comparison["GrLivArea"] - defaults["GrLivArea"]).abs()
        )
        comps = comparison.sort_values("distance").head(8).drop(columns="distance")
        st.dataframe(
            comps.rename(
                columns={
                    TARGET: "Sale Price",
                    "Neighborhood": "Neighborhood",
                    "OverallQual": "Quality",
                    "GrLivArea": "Living Area",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

with tab_batch:
    st.subheader("Upload CSV For Batch Valuation")
    uploaded_file = st.file_uploader("CSV must follow the training/test feature schema.", type=["csv"])
    if uploaded_file:
        batch = pd.read_csv(uploaded_file)
        batch_predictions = predict_prices(model, batch)
        scored = batch.copy()
        scored["PredictedSalePrice"] = batch_predictions
        st.success(f"Scored {len(scored):,} properties.")
        st.dataframe(scored.head(50), use_container_width=True)
        st.download_button(
            "Download scored CSV",
            data=scored.to_csv(index=False),
            file_name="scored_house_prices.csv",
            mime="text/csv",
        )
    else:
        st.info("Upload a property CSV to generate client-ready estimates.")

with tab_insights:
    st.subheader("Market Snapshot")
    c1, c2 = st.columns(2)
    with c1:
        neighborhood = (
            train_df.groupby("Neighborhood")[TARGET]
            .median()
            .sort_values(ascending=False)
            .head(15)
        )
        st.bar_chart(neighborhood)
    with c2:
        quality = train_df.groupby("OverallQual")[TARGET].median()
        st.line_chart(quality)

    st.dataframe(
        train_df[["Neighborhood", "OverallQual", "GrLivArea", "GarageCars", TARGET]]
        .sort_values(TARGET, ascending=False)
        .head(25),
        use_container_width=True,
        hide_index=True,
    )
