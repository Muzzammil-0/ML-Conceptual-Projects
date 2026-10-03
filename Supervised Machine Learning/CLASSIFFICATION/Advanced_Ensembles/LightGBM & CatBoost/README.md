# Advanced Ensembles — XGBoost vs LightGBM vs CatBoost

Comparing the three major gradient boosting frameworks on the Telco Customer Churn dataset, focusing on native categorical handling, training speed, and the performance trade-offs between them.

## Dataset
- Telco Customer Churn (~7,000 rows)
- 19 raw features (including 16 categorical columns)
- Target: Churn (binary)

## Approach
1. **XGBoost:** Standard pre-processing (One-Hot Encoding). Input expanded to 30 numeric features.
2. **LightGBM:** Passed raw DataFrame, casting object columns to `category` dtype for native handling.
3. **CatBoost:** Passed raw DataFrame, explicitly declaring categorical feature indices.

## Results

| Model | F1 Score | Train Time | Input Features | Notes |
| :--- | :--- | :--- | :--- | :--- |
| XGBoost | 0.6081 | 0.53 sec | 30 | Fastest, needs manual OHE. |
| LightGBM | 0.6132 | 6.48 sec | 19 | Leaf-wise growth, native categoricals. |
| **CatBoost** | **0.6203** | 19.89 sec | 19 | **Highest F1.** Best categorical handling. |

### CatBoost Top Features
- `Contract` (22.3%)
- `tenure` (14.3%)
- `TotalCharges` (10.7%)
- `MonthlyCharges` (10.5%)
- `InternetService` (7.6%)

## Key Observations
- **CatBoost won on accuracy.** Its "Ordered Boosting" algorithm handles categorical variables mathematically better than One-Hot Encoding, extracting more signal from `Contract` and `InternetService`.
- **The speed paradox.** On a tiny 7k-row dataset, XGBoost is fastest because it has zero categorical-conversion overhead. LightGBM's category-dtype conversion and CatBoost's heavy target encoding math make them slower here. On datasets >1M rows, LightGBM becomes the fastest by a massive margin.
- **The tree-growth difference.** LightGBM uses leaf-wise growth (splitting the most promising leaf), which is faster and more accurate but prone to overfitting on small data. XGBoost and CatBoost use level-wise growth, which is more stable.
- **Feature importance consistency.** Despite entirely different mathematical approaches, CatBoost's top 5 features are identical to the SHAP analysis performed on XGBoost (`Contract`, `tenure`, `TotalCharges`, `MonthlyCharges`, `InternetService`). This confirms those are the true drivers of churn in this dataset.

## Stack
`xgboost` · `lightgbm` · `catboost` · `pandas` · `scikit-learn`