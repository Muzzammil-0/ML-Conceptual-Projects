# Linear Models — L1 (Lasso) vs L2 (Ridge) Regularization

Demonstration of how regularization affects feature selection and multicollinearity using Logistic Regression on the Telco Customer Churn dataset.

## Dataset
- Telco Customer Churn (34 engineered features)
- Preprocessing: `StandardScaler` (mandatory for regularized linear models)
- Class imbalance handled via `class_weight='balanced'`

## Approach
1. Train a Logistic Regression model with L2 (Ridge) regularization.
2. Train a Logistic Regression model with L1 (Lasso) regularization.
3. Compare the number of non-zero coefficients (sparsity).
4. Inspect how each model handles the redundant engineered features (`TotalCharges` vs `AvgMonthlyCharge`).
5. Visualize the top 10 churn drivers based on L1 coefficients.

## Results
- **Total Features:** 34
- **L2 Non-zero Coefficients:** 34
- **L1 Non-zero Coefficients:** 32 (L1 dropped 2 features)
- **L2 Test F1:** 0.6076
- **L1 Test F1:** 0.6076

### How the models handled redundancy:
| Feature | L2 (Ridge) Weight | L1 (Lasso) Weight |
| :--- | :--- | :--- |
| `AvgMonthlyCharge` | 0.004 | **0.000** (Dropped) |
| `ChargeToTenureRatio` | 0.345 | 0.341 |
| `TotalCharges` | 0.391 | 0.369 |
| `MonthlyCharges` | -1.055 | -0.648 |

## Key Observations
- **L1 performs automatic feature selection.** By forcing weights to exactly zero, L1 completely eliminated the redundant `AvgMonthlyCharge` feature, whereas L2 just shrunk it to a negligible non-zero value.
- **Collinearity resolution.** When features are perfectly correlated, L1 arbitrarily picks one to keep and zeroes out the other. L2 splits the weight between them.
- **Equal performance, better simplicity.** The F1 score remained identical (0.6076) for both models, proving that dropping those features (L1) did not cost any predictive power. L1 simply resulted in a simpler, more interpretable model.
- **Interpretability.** Linear models give signed coefficients (log-odds). Positive weights (e.g., Month-to-month contracts) increase churn probability, while negative weights (e.g., Two-year contracts) decrease it.

## Stack
`scikit-learn` · `pandas` · `numpy` · `matplotlib`