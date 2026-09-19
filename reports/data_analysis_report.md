# House Price Data Analysis Report

## Executive Summary
- Dataset used: `data.csv` with 1,460 properties and 80 explanatory columns.
- Target variable: `SalePrice`, ranging from $34,900 to $755,000; median price is $163,000.
- Average sale price is $180,921, which is higher than the median, so the market is right-skewed by expensive homes.
- The strongest numeric price relationships are led by OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF.

## Data Quality
- Numeric columns: 38. Categorical columns: 43.
- Columns with missing values: 19. Most missing fields represent optional amenities such as pool, alley, fence, fireplace, garage, and basement attributes.
- The model pipeline handles missing values with median imputation for numeric fields and explicit missing categories for categorical fields.

### Highest Missing Value Rates

| Feature | Missing Count | Missing Rate |
| --- | ---: | ---: |
| PoolQC | 1,453 | 99.5% |
| MiscFeature | 1,406 | 96.3% |
| Alley | 1,369 | 93.8% |
| Fence | 1,179 | 80.8% |
| MasVnrType | 872 | 59.7% |
| FireplaceQu | 690 | 47.3% |
| LotFrontage | 259 | 17.7% |
| GarageType | 81 | 5.5% |
| GarageYrBlt | 81 | 5.5% |
| GarageFinish | 81 | 5.5% |
| GarageQual | 81 | 5.5% |
| GarageCond | 81 | 5.5% |

## Price Distribution
- 25th percentile: $129,975.
- Median: $163,000.
- 75th percentile: $214,000.
- 90th percentile: $278,000.

## Feature Relationships

| Feature | Correlation With Price | Interpretation |
| --- | ---: | --- |
| OverallQual | 0.791 | Higher material and finish quality strongly increases expected price. |
| GrLivArea | 0.709 | Larger above-grade living area raises price substantially. |
| GarageCars | 0.640 | More garage capacity is associated with higher-value homes. |
| GarageArea | 0.623 | Larger garage area tracks with higher prices. |
| TotalBsmtSF | 0.614 | More basement area generally increases valuation. |
| 1stFlrSF | 0.606 | Larger first-floor area is positively valued. |
| FullBath | 0.561 | More full bathrooms support higher valuations. |
| TotRmsAbvGrd | 0.534 | More rooms above grade usually reflect larger and more valuable homes. |
| YearBuilt | 0.523 | Newer homes generally sell at higher prices. |
| YearRemodAdd | 0.507 | More recent remodels are associated with stronger sale prices. |
| GarageYrBlt | 0.486 | Meaningful numeric association with sale price. |
| MasVnrArea | 0.477 | Meaningful numeric association with sale price. |

## Area Analysis

| Neighborhood | Homes | Median Price | Median Living Area | Median Quality | Median Price Per Sq Ft |
| --- | ---: | ---: | ---: | ---: | ---: |
| NridgHt | 77 | $315,000 | 1,850 | 8.0 | $170 |
| NoRidge | 41 | $301,500 | 2,418 | 8.0 | $125 |
| StoneBr | 25 | $278,000 | 1,742 | 8.0 | $160 |
| Timber | 38 | $228,475 | 1,690 | 7.5 | $135 |
| Somerst | 86 | $225,500 | 1,564 | 7.0 | $144 |
| Veenker | 11 | $218,000 | 1,437 | 6.0 | $152 |
| Crawfor | 51 | $200,624 | 1,717 | 6.0 | $117 |
| ClearCr | 28 | $200,250 | 1,738 | 6.0 | $115 |
| CollgCr | 150 | $197,200 | 1,500 | 7.0 | $131 |
| Blmngtn | 17 | $191,000 | 1,500 | 7.0 | $127 |
| NWAmes | 73 | $182,900 | 1,664 | 6.0 | $110 |
| Gilbert | 79 | $181,000 | 1,593 | 7.0 | $114 |
| SawyerW | 59 | $179,900 | 1,603 | 6.0 | $112 |
| Mitchel | 49 | $153,500 | 1,204 | 5.0 | $127 |
| NPkVill | 9 | $146,000 | 1,322 | 6.0 | $110 |

## Machine Learning Model
- Algorithm: log-target ensemble regression combining histogram gradient boosting and random forest inside a scikit-learn pipeline.
- Preprocessing: generated housing features, numeric imputation, robust scaling, categorical missing handling, and one-hot encoding.
- Validation RMSLE: 0.1454.
- Validation MAE: $15,778.
- Validation R2: 0.875.
- Cross-validation RMSLE mean: 0.1290.

## Buying Suggestions
- Prioritize `OverallQual`, `GrLivArea`, `TotalBsmtSF`, garage capacity, and neighborhood before smaller cosmetic attributes because these show the strongest relationship with price.
- For budget-sensitive buyers, target neighborhoods with below-median prices but acceptable living area and quality rather than simply choosing the cheapest property.
- For family buyers, shortlist homes with at least median living area, adequate bedrooms and bathrooms, and garage capacity before optimizing for extras such as porch, deck, or pool.
- For premium buyers, pay attention to high-quality neighborhoods and newer or recently remodeled homes; the data shows quality and age-related variables carry large pricing power.

### Customer Shortlist By Area And Price

| Segment | Neighborhood | Median Price | Median Living Area | Median Quality | Price Per Sq Ft |
| --- | --- | ---: | ---: | ---: | ---: |
| Value buy | SWISU | $139,500 | 1,691 | 6.0 | $82 |
| Budget friendly | MeadowV | $88,000 | 1,092 | 4.0 | $81 |
| Budget friendly | OldTown | $119,000 | 1,374 | 5.0 | $87 |
| Budget friendly | IDOTRR | $103,000 | 1,128 | 5.0 | $91 |
| Budget friendly | BrDale | $106,000 | 1,155 | 6.0 | $92 |
| Budget friendly | Edwards | $121,750 | 1,200 | 5.0 | $101 |
| Budget friendly | BrkSide | $124,300 | 1,210 | 5.0 | $103 |
| Budget friendly | NAmes | $140,000 | 1,200 | 5.0 | $117 |
| Budget friendly | Sawyer | $135,000 | 1,106 | 5.0 | $122 |
| Balanced shortlist | NWAmes | $182,900 | 1,664 | 6.0 | $110 |
| Balanced shortlist | SawyerW | $179,900 | 1,603 | 6.0 | $112 |
| Balanced shortlist | ClearCr | $200,250 | 1,738 | 6.0 | $115 |

## Deliverables Generated
- Model artifact: `artifacts/house_price_model.joblib`.
- Model metrics: `artifacts/metrics.json`.
- Validation audit: `reports/validation_predictions.csv`.
- Feature profile for the app: `artifacts/feature_profile.json` with 80 model input columns.
