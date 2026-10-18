import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc, precision_recall_curve,
                             classification_report)
from xgboost import XGBClassifier

# ---------- 1. Load Data (Direct CSV fallback) ----------
import pandas as pd

# Bypass OpenML entirely to avoid network timeouts
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

y = (df['Churn'] == 'Yes').astype(int)
X = pd.get_dummies(df.drop(columns=['Churn']), drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------- 2. Train Baseline Model ----------
# We use the same XGBoost setup from before
xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, 
                    eval_metric='logloss', early_stopping_rounds=30, random_state=42)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Get PROBABILITIES, not hard labels (0 or 1)
y_probs = xgb.predict_proba(X_test)[:, 1] 
y_pred_default = (y_probs >= 0.5).astype(int)

print("=== BASELINE (Threshold = 0.5) ===")
print(classification_report(y_test, y_pred_default, target_names=['No Churn', 'Churn']))

# ---------- 3. ROC & PR Curves ----------
fpr, tpr, roc_thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

precision, recall, pr_thresholds = precision_recall_curve(y_test, y_probs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate (Recall)')
axes[0].set_title('ROC Curve')
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

# PR Curve
axes[1].plot(recall, precision, color='green', lw=2)
axes[1].set_xlabel('Recall (Churn)')
axes[1].set_ylabel('Precision (Churn)')
axes[1].set_title('Precision-Recall Curve')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ---------- 4. Threshold Tuning ----------
thresholds = np.arange(0.1, 0.9, 0.05)
results = []

for t in thresholds:
    y_pred_t = (y_probs >= t).astype(int)
    results.append({
        'threshold': t,
        'precision': precision_score(y_test, y_pred_t),
        'recall': recall_score(y_test, y_pred_t),
        'f1': f1_score(y_test, y_pred_t)
    })

results_df = pd.DataFrame(results)

plt.figure(figsize=(8, 5))
plt.plot(results_df['threshold'], results_df['precision'], label='Precision', marker='o')
plt.plot(results_df['threshold'], results_df['recall'], label='Recall', marker='s')
plt.plot(results_df['threshold'], results_df['f1'], label='F1 Score', marker='^')
plt.axvline(0.5, color='gray', linestyle='--', label='Default (0.5)')
plt.xlabel('Decision Threshold')
plt.ylabel('Score')
plt.title('Threshold Tuning: Precision vs Recall vs F1')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ---------- 5. Apply a Custom Threshold ----------
# If we want to catch ~80% of churners, what threshold do we need?
# Let's find the threshold that gives at least 0.80 recall
target_recall = 0.80
valid_t = results_df[results_df['recall'] >= target_recall]

if not valid_t.empty:
    best_t = valid_t.loc[valid_t['precision'].idxmax()]['threshold']
    print(f"\n=== CUSTOM THRESHOLD (t = {best_t:.2f}) ===")
    print(f"To achieve >= {target_recall*100}% Recall, we accept Precision = {valid_t.loc[valid_t['precision'].idxmax()]['precision']:.3f}")
    
    y_pred_custom = (y_probs >= best_t).astype(int)
    print(classification_report(y_test, y_pred_custom, target_names=['No Churn', 'Churn']))
else:
    print("\nCould not achieve target recall at these thresholds.")

# ---------- 6. Class Weights (The Alternative to Thresholds) ----------
# Instead of tuning thresholds post-training, we tell the model DURING training 
# that churners are more important. 73% no-churn / 27% churn -> ratio ~ 2.7
scale_weight = (len(y_train) - sum(y_train)) / sum(y_train)

xgb_weighted = XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=5, 
    scale_pos_weight=scale_weight,  # <--- The magic parameter
    eval_metric='logloss', early_stopping_rounds=30, random_state=42
)
xgb_weighted.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred_weighted = xgb_weighted.predict(X_test)

print(f"\n=== CLASS WEIGHTS (scale_pos_weight = {scale_weight:.2f}) ===")
print(classification_report(y_test, y_pred_weighted, target_names=['No Churn', 'Churn']))