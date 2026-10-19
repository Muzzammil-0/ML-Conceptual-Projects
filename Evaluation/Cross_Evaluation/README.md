# Cross-Validation & Model Stability

Evaluating the variance and stability of an XGBoost classifier on the Telco Customer Churn dataset using Stratified K-Fold Cross-Validation.

## Dataset
- Telco Customer Churn (IBM GitHub CSV)
- Binary target, ~73/27 class imbalance.
- XGBoost with `scale_pos_weight` (2.76) to handle the imbalance.

## Approach
1. Train/test split (80/20) to hold out a final unbiased test set.
2. Train baseline model and evaluate on the single test split.
3. Run Stratified 5-Fold Cross-Validation on the training set.
4. Track Accuracy, Precision, Recall, F1, and ROC-AUC across all 5 folds.
5. Compute mean and standard deviation for each metric.
6. Visualize metric spread using a boxplot.

## Results

### Single Split vs. Cross-Validation
- **Single Split Accuracy:** 0.7370
- **Single Split Recall:** 0.7674

### 5-Fold Cross-Validation Results
- **Accuracy:** 0.7657 +/- 0.0064 
- **Precision:** 0.5426 +/- 0.0087
- **Recall:** 0.7545 +/- 0.0209
- **F1 Score:** 0.6312 +/- 0.0114
- **ROC-AUC:** 0.8423 +/- 0.0051

## Key Observations
- **Single-split evaluations can be misleading.** The single train/test split yielded an accuracy of 0.7370, but 5-fold CV revealed the true expected performance is 0.7657. The single split happened to draw a slightly harder test set.
- **The model is highly stable.** An accuracy standard deviation of `+/- 0.0064` means the model's performance does not fluctuate wildly depending on which users land in the test set.
- **Recall is the most volatile metric.** With a std of `+/- 0.0209`, the exact recall score is more sensitive to data splits than overall accuracy or ROC-AUC, which is expected when using class weights that push predictions closer to the decision boundary.
- **Stratification matters.** With a 27% minority class, random K-Fold might create folds with wildly different churn rates. `StratifiedKFold` guarantees each fold maintains the 73/27 ratio.

## Stack
`scikit-learn` · `xgboost` · `pandas` · `matplotlib`