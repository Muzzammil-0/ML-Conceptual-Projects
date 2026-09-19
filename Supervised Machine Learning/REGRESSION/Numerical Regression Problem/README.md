# Regression — Predicting House Prices

A comparison of Linear Regression, Random Forest, and XGBoost for tabular regression on the California Housing dataset, demonstrating the necessity of non-linear models for spatial data.

## Dataset
- `fetch_california_housing` (20,640 rows, 8 features)
- Target: Median House Value (in $100,000s), ranging from $15,000 to $500,000.

## Approach
1. Split data into 80% train / 20% test.
2. Train three models: Linear Regression, Random Forest (100 trees), and XGBoost (300 estimators).
3. Evaluate using regression metrics: RMSE, MAE, and R².
4. Plot Actual vs. Predicted values for the best model to observe error distribution.

## Results

| Model | RMSE | MAE | R² |
| :--- | :--- | :--- | :--- |
| Linear Regression | $74,558 | $53,320 | 0.5758 |
| Random Forest | $50,534 | $32,754 | 0.8051 |
| **XGBoost** | **$46,697** | **$30,698** | **0.8336** |

## Key Observations
- **Non-linearity is mandatory for spatial data.** Linear Regression performs poorly (R² 0.58) because house prices do not scale linearly with location. Trees capture the complex interaction between Latitude/Longitude and price.
- **RMSE vs. MAE gap.** XGBoost's RMSE ($46k) is significantly higher than its MAE ($30k). This indicates the model makes mostly small errors but occasionally makes massive errors, likely on the ultra-expensive homes.
- **The Ceiling Effect.** The dataset artificially caps house prices at $500,000 (5.0). In the Actual vs. Predicted plot, models will systematically under-predict these expensive homes because there is a hard limit on the target variable, and the models cannot extrapolate beyond their training bounds.

## Stack
`scikit-learn` · `xgboost` · `pandas` · `numpy` · `matplotlib`