# Advanced House Price Prediction

Client-style data analysis and machine learning project for predicting residential sale prices from the supplied Ames Housing dataset.

## What Is Included

- Complete data analysis report with feature relationships, area analysis, model metrics, and customer buying suggestions.
- End-to-end scikit-learn training pipeline with imputation, encoding, robust scaling, feature engineering, ensemble models, and log-target regression.
- Saved production artifact at `artifacts/house_price_model.joblib`.
- Validation metrics and prediction audit files.
- Streamlit dashboard for single-property valuation, batch CSV scoring, and market insights.
- Command-line scripts for retraining and generating submission-style predictions.

## Project Structure

```text
.
+-- app/streamlit_app.py
+-- artifacts/
+-- data/
|   +-- data.csv
+-- reports/
+-- src/
|   +-- config.py
|   +-- data.py
|   +-- features.py
|   +-- model.py
|   +-- predict.py
|   +-- train.py
+-- submissions/
+-- requirements.txt
+-- README.md
```

## Quick Start

```bash
python3 -m src.train
python3 -m src.report
python3 -m src.predict
streamlit run app/streamlit_app.py
```

## Model Approach

The model uses a production-style pipeline so training, validation, web predictions, and batch scoring all share the same transformations. It combines:

- Domain features such as total square footage, house age, remodel age, total bathrooms, and amenity flags.
- Numeric median imputation and robust scaling.
- Categorical missing-value handling and one-hot encoding with rare-category grouping.
- A stacked ensemble using gradient boosting, random forest, and ridge meta-regression.
- Log-transformed target learning to optimize behavior on expensive homes and reduce skew.

## Outputs

After training, these files are generated:

- `artifacts/house_price_model.joblib`: trained model pipeline.
- `artifacts/metrics.json`: validation and cross-validation scores.
- `artifacts/feature_profile.json`: UI defaults and allowed categorical values.
- `reports/data_analysis_report.md`: complete analysis report and customer recommendation guide.
- `reports/validation_predictions.csv`: validation predictions sorted by largest error.
- `submissions/scored_house_prices.csv`: scored predictions for the supplied CSV.

## Client Notes

This is designed as a strong portfolio/client deliverable rather than a notebook-only demo. The dashboard supports executive-friendly valuation, analyst batch scoring, and quick market exploration from one interface.
