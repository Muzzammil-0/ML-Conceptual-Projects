# Model Evaluation — Handling Class Imbalance & Thresholds

Analysis of classification metrics, decision thresholds, and class weights on the Telco Customer Churn dataset (~7k rows, 27% positive class). Demonstrates why default accuracy and a 0.5 threshold are misleading for imbalanced data.

## Dataset
- Telco Customer Churn (`Telco-Customer-Churn.csv`)
- 20 features, binary target (`Churn`), 73/27 class imbalance
- Standard preprocessing: `TotalCharges` coerced to numeric, categoricals one-hot encoded.

## Approach
1. Train baseline XGBoost model.
2. Evaluate at default threshold (0.5).
3. Plot ROC and Precision-Recall (PR) curves.
4. Sweep decision thresholds from 0.1 to 0.9 to observe the precision-recall tradeoff.
5. Apply a custom threshold to achieve ≥ 80% recall.
6. Train a second XGBoost model using `scale_pos_weight` (ratio ≈ 2.76) and compare.

## Results

| Model / Strategy | Threshold | Accuracy | Churn Precision | Churn Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline XGB** | 0.50 | 0.80 | 0.64 | 0.52 |
| **Threshold Tuned** | 0.25 | 0.71 | 0.48 | 0.82 |
| **Class Weights** | 0.50 | 0.73 | 0.50 | 0.80 |

## Key Observations
- **Accuracy is a misleading metric for imbalanced data.** The baseline model had the highest accuracy (0.80) but missed 48% of actual churners.
- **The 0.5 threshold is arbitrary.** Lowering the threshold to 0.25 increased recall from 0.52 to 0.82, sacrificing precision.
- **Class weights simulate threshold tuning.** `scale_pos_weight` achieved nearly identical recall/precision tradeoffs by altering the training loss rather than the decision boundary.
- **Business Translation:** In a retention campaign, a false positive (wasted offer) is cheap, while a false negative (lost customer) is expensive. The tuned models are strictly superior for the business objective despite lower overall accuracy.

## Stack
`scikit-learn` · `xgboost` · `pandas` · `matplotlib`