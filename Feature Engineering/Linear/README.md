# Feature Engineering — Impact on Linear Models (Logistic Regression)

Testing the effect of handcrafted ratio and interaction features on a Logistic Regression classifier.

## Dataset
- Telco Customer Churn (IBM GitHub CSV)
- Preprocessing: `StandardScaler`, `class_weight='balanced'`
- Engineered Features: `AvgMonthlyCharge`, `NumServices`, `ChargeToTenureRatio`, `IsNewCustomer`

## Results
- **Raw Features F1:** 0.6075
- **Engineered Features F1:** 0.6070
- **Delta:** -0.0005 (No improvement)

## Key Observations
- **Multicollinearity hurts linear models.** Feeding the model `TotalCharges`, `tenure`, *and* `TotalCharges/tenure` creates perfect collinearity. The model struggles to assign weights, and L2 regularization shrinks the coefficients of the redundant features.
- **Categorical dominance.** The primary churn signal in this dataset comes from categorical features (like `Contract_Month-to-month`), not continuous features. Creating ratios of continuous features doesn't help if the categorical features are already doing the heavy lifting.
- **Empirical validation beats theoretical intuition.** While linear models *theoretically* benefit from ratio features, the specific structure of this dataset means the new features just added noise and collinearity.
- **Conclusion:** Feature engineering must be domain-driven. Blindly applying mathematical transformations to existing columns without adding new external context often fails to improve the score.

## Stack
`scikit-learn` · `pandas`