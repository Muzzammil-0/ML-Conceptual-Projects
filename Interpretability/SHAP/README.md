# SHAP — Model Interpretability for XGBoost

Using SHapley Additive exPlanations (SHAP) to explain the predictions of a tuned XGBoost classifier on the Telco Customer Churn dataset. Bridging the gap between black-box performance and linear model transparency.

## Dataset
- Telco Customer Churn
- Tuned XGBoost model (depth=4, lr=0.016, scale_pos_weight=2.76)

## Approach
1. Train the tuned XGBoost model.
2. Initialize `shap.TreeExplainer` and calculate SHAP values for the test set using the modern `Explanation` API (`explainer(X_test)`).
3. **Global Interpretability:** Generate a SHAP summary plot (beeswarm) to rank the most impactful features across the entire dataset.
4. **Local Interpretability:** Generate a SHAP waterfall plot to explain a single high-risk customer's prediction (Customer Index 1149, 95% churn prob).
5. **Feature Interactions:** Generate dependence plots to see how `tenure` and `InternetService_Fiber optic` interact.

## Results

### Global Insights (Summary Plot)
- **Top Churn Drivers:** `Contract_Month-to-month`, `InternetService_Fiber optic`, and `tenure` (low tenure increases churn).
- **Top Churn Mitigators:** `Contract_Two year` and high `tenure`.
- **Feature Direction:** High values (red) of month-to-month contracts push churn risk up. High values (red) of tenure push churn risk down. 

### Local Insights (Waterfall Plot)
- Explained exactly why Customer 1149 has a 95% churn probability.
- The model decomposed the prediction, showing exactly how much each feature (e.g., missing tech support, high monthly charges) pushed the probability higher, starting from the dataset's baseline average.

### Dependence Plots (Interaction)
- **The Tenure Cliff:** Churn risk drops sharply after the 12-month mark.
- **Fiber Optic Interaction:** New Fiber Optic customers have a significantly higher churn risk than new DSL customers, indicating a potential product-specific onboarding issue.

## Key Observations
- **SHAP bridges the interpretability gap.** It gives exact "weights" for a specific prediction, regardless of how complex the model is, mimicking the transparency of Logistic Regression while keeping the power of XGBoost.
- **Directional Impact.** Unlike traditional `feature_importances_` (which only tells you *how often* a feature is used), SHAP tells you *how much* and *in which direction* it impacted the final prediction.
- **Actionable Business Value.** Handing the waterfall plot to a retention team provides exact, personalized reasons for churn ("This customer is leaving because they are on a month-to-month fiber-optic plan without tech support"), making it actionable.

## Stack
`xgboost` · `shap` · `pandas` · `matplotlib` · `numpy`