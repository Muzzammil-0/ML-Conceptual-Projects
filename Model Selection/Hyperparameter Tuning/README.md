# Hyperparameter Tuning — GridSearch vs. RandomizedSearch

Systematic hyperparameter optimization of an XGBoost classifier on the Telco Customer Churn dataset. Compares exhaustive grid search against randomized sampling.

## Dataset
- Telco Customer Churn (IBM GitHub CSV)
- Optimized metric: F1-Score on the minority class (Churn)
- Cross-validation: 3-Fold inside the search, final holdout on 20% test set
- Base model includes `scale_pos_weight=2.76` to handle class imbalance

## Approach
1. Establish a baseline XGBoost model with manually chosen parameters.
2. Run `GridSearchCV` across a discrete parameter grid (18 combinations).
3. Run `RandomizedSearchCV` across wider distributions (20 iterations, including `subsample` and `colsample_bytree`).
4. Evaluate the best models from both searches on the unseen test set.

## Results

| Model | Parameters / Strategy | CV F1 | Test F1 | Fit Time |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Manual (depth=5, lr=0.05) | — | 0.6081 | — |
| **Grid Search** | depth=7, lr=0.05, n=100 | 0.6340 | 0.6089 | 20.2s |
| **Random Search** | depth=4, lr=0.016, n=317, subsample=0.78 | **0.6384** | **0.6223** | **10.8s** |

## Key Observations
- **Randomized Search is vastly more efficient.** It found a better model in half the time. Grid Search wastes compute on combinations that don't matter (e.g., deep trees with slow learning rates).
- **"Slow and steady" wins.** Random Search found that shallower trees (`max_depth=4`) with a slower learning rate (`0.016`) and more estimators (`317`) generalize better than deep trees with fast learning rates.
- **The Tuning Trap is visible.** The CV F1 (0.6384) is slightly higher than the final Test F1 (0.6223). Searching over many parameters can cause the model to slightly overfit the internal validation folds; the holdout test set reveals the true unbiased performance.
- **Imbalance handling persists.** The best tuned model maintains a high Churn Recall (0.81) at the cost of Precision (0.50) because `scale_pos_weight` was kept fixed during tuning.

## Stack
`scikit-learn` · `xgboost` · `scipy` · `pandas`