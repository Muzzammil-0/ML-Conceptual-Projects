# Polynomial Features — The Curse of Dimensionality

Using `PolynomialFeatures` to allow a linear model (Logistic Regression) to learn non-linear decision boundaries, demonstrating the exponential feature explosion and its impact on generalization.

## Dataset
- Telco Customer Churn (30 encoded features)
- Dropped `customerID` to prevent factorial memory explosion during expansion.

## Approach
1. **2D Visualization:** Trained Logistic Regression on just `Tenure` and `MonthlyCharges` with Degree 1 (linear) and Degree 3 (polynomial). Plotted decision boundaries to show how polynomial features allow straight lines to bend.
2. **High-Dimensional Test:** Trained models on the full tabular dataset at Degree 1, Degree 2, and Degree 3 to observe feature explosion and test performance.

## Results

| Model | Number of Features | Test F1 (Churn) |
| :--- | :--- | :--- |
| Baseline (Degree 1) | 30 | 0.6075 |
| Polynomial (Degree 2) | 495 | 0.6082 |
| Polynomial (Degree 3) | 5,455 | **0.5534** |

## Key Observations
- **Linear models can learn curves.** The 2D visualization proves that by creating squared and interaction terms, a linear model can draw complex, curved decision boundaries in the original feature space.
- **The combinatorial explosion.** The number of features grows factorially. A dataset with 30 features becomes 495 features at Degree 2, and 5,455 features at Degree 3.
- **The Curse of Dimensionality.** The Degree 3 model performed *significantly worse* than the baseline (0.5534 vs 0.6075). With 5,455 features and ~5,600 training samples, the model memorized the noise in the training data (overfitting) and failed to generalize. 
- **Why Trees Win.** This experiment explains why tree-based models (XGBoost, Random Forest) dominate tabular ML. Trees can approximate non-linear curves natively without requiring you to manually create thousands of polynomial features.
- **Practical Takeaway:** Polynomial features are useful only for low-dimensional data (2-5 features) or when used with heavy L1 regularization to kill the 5,000 useless interaction terms.

## Stack
`scikit-learn` · `pandas` · `numpy` · `matplotlib`