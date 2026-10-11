# Feature Engineering — Impact on Tree-Based Models (XGBoost)

Testing the effect of handcrafted ratio and interaction features on an XGBoost classifier.

## Dataset
- Telco Customer Churn (IBM GitHub CSV)
- Engineered Features: `AvgMonthlyCharge`, `NumServices`, `ChargeToTenureRatio`, `IsNewCustomer`

## Results
- **Raw Features F1:** 0.6196
- **Engineered Features F1:** 0.6206
- **Delta:** +0.001 (Statistically insignificant)

## Key Observations
- **Trees are invariant to monotonic transformations.** XGBoost builds decision boundaries by splitting on single features (e.g., `tenure < 12`). It approximates non-linear relationships internally, so it doesn't need pre-computed ratios.
- **Multicollinearity is harmless to trees.** Adding `TotalCharges/tenure` alongside `TotalCharges` and `tenure` doesn't hurt XGBoost (it just picks the best split), but it also doesn't help it.
- **Conclusion:** For tree ensembles, feature engineering should focus on *external* data or aggregations across rows, not mathematical transformations of existing columns.

## Stack
`xgboost` · `pandas` · `scikit-learn`