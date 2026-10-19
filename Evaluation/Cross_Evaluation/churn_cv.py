import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from xgboost import XGBClassifier

# ---------- 1. Load Data ----------
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

y = (df['Churn'] == 'Yes').astype(int)
X = pd.get_dummies(df.drop(columns=['Churn']), drop_first=True)

# We still need a holdout test set for the FINAL evaluation. 
# CV is for model selection/tuning. The test set is for the final unbiased check.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------- 2. Setup Model & Metrics ----------
# scale_pos_weight used to fix the imbalance (from the previous file)
model = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    scale_pos_weight=2.76, eval_metric='logloss', random_state=42
)

# We want to track multiple metrics, not just accuracy
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# ---------- 3. Single Train/Test Split (What we did before) ----------
model.fit(X_train, y_train)
y_pred_single = model.predict(X_test)

from sklearn.metrics import accuracy_score, recall_score
print("=== SINGLE TRAIN/TEST SPLIT ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_single):.4f}")
print(f"Churn Recall: {recall_score(y_test, y_pred_single):.4f}")

# ---------- 4. Stratified 5-Fold Cross-Validation ----------
# Stratified ensures each fold has the same 73/27 class ratio
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    model, X_train, y_train, 
    cv=cv, 
    scoring=scoring_metrics, 
    return_train_score=False
)

print("\n=== 5-FOLD CROSS-VALIDATION ===")
for metric in scoring_metrics:
    scores = cv_results[f'test_{metric}']
    print(f"{metric.capitalize():10s} : {scores.mean():.4f} +/- {scores.std():.4f}  |  folds: {np.round(scores, 4)}")

# ---------- 5. Visualize the Variance ----------
# A boxplot shows the spread of the 5 folds for each metric
fig, ax = plt.subplots(figsize=(8, 5))
data_to_plot = [cv_results[f'test_{m}'] for m in scoring_metrics]
ax.boxplot(data_to_plot, labels=scoring_metrics, patch_artist=True)
ax.set_title("Cross-Validation Stability (5 Folds)")
ax.set_ylabel("Score")
ax.grid(alpha=0.3)
plt.show()