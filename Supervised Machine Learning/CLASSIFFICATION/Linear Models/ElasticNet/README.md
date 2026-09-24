# Linear Models — ElasticNet Regularization

Implementation of ElasticNet (combined L1/L2 regularization) on the Telco Customer Churn dataset, testing theoretical expectations against empirical results on highly collinear features.

## Dataset
- Telco Customer Churn (32 features)
- Engineered features include mathematically derived collinear columns (`AvgMonthlyCharge = TotalCharges / (tenure + 1)`).

## Approach
1. Train Logistic Regression models using the `saga` solver with varying `l1_ratio` values:
   - Pure Lasso (`l1_ratio=1.0`)
   - ElasticNet 50/50 (`l1_ratio=0.5`)
   - ElasticNet 80/20 (`l1_ratio=0.8`)
   - Pure Ridge (`l1_ratio=0.0`)
2. Observe the coefficients of the collinear feature group across models.
3. Compare the Test F1 score across all regularization strategies.

## Results

| Model | Features Kept | `AvgMonthlyCharge` Weight | Test F1 |
| :--- | :--- | :--- | :--- |
| Pure Lasso | 31 / 32 | 0.000 | 0.6023 |
| ElasticNet 50/50 | 31 / 32 | 0.000 | 0.6012 |
| ElasticNet 80/20 | 31 / 32 | 0.000 | 0.6010 |
| Pure Ridge | 32 / 32 | -0.032 | 0.6033 |

## Key Observations
- **Theory vs. Reality:** The "grouping effect" of ElasticNet did not keep `AvgMonthlyCharge` alive. Because the feature is mathematically derived from existing features (perfect collinearity), even a 50/50 L1/L2 split was aggressive enough to zero it out.
- **L1 does the feature selection.** Whether pure Lasso or ElasticNet, any model with an L1 penalty component dropped the redundant feature. Only Ridge kept it.
- **The linear ceiling.** The F1 score hovered around ~0.60 for all four models. Once a linear model learns the main drivers, changing the penalty term changes *which* features it keeps, but not the maximum predictive power it can achieve.
- **Practical takeaway:** ElasticNet is a safety net. If you don't know whether your data has many correlated features (Lasso might drop one arbitrarily) or many useful features (Ridge might keep everything), ElasticNet offers a stable middle ground.

## Stack
`scikit-learn` · `pandas` · `numpy`