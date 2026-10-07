import numpy as np
import pandas as pd
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

# ---------- 1. Load Data ----------
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

y = (df['Churn'] == 'Yes').astype(int)
X = pd.get_dummies(df.drop(columns=['Churn']), drop_first=True)

# 80/20 split. We tune on the 80%, and test on the 20%.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Base model setup (we fix scale_pos_weight to focus on tree params)
base_model = XGBClassifier(
    scale_pos_weight=2.76, 
    eval_metric='logloss', 
    random_state=42
)

# ---------- 2. Define the Search Space ----------
# For GridSearch, we need a small dictionary of discrete values
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300]
}

# For RandomizedSearch, we can use distributions
param_dist = {
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.2),
    'n_estimators': randint(100, 600),
    'subsample': uniform(0.6, 0.4),        # 0.6 to 1.0
    'colsample_bytree': uniform(0.6, 0.4)  # 0.6 to 1.0
}

# ---------- 3. GridSearchCV ----------
print("=== STARTING GRID SEARCH ===")
start = time.time()

# We optimize for F1-score (balance of precision/recall)
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,                    # 3-fold CV internally
    scoring='f1',            # Optimize F1 on the churn class
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

grid_time = time.time() - start
print(f"GridSearch completed in {grid_time:.1f} seconds.")
print(f"Best Grid Params: {grid_search.best_params_}")
print(f"Best Grid CV F1: {grid_search.best_score_:.4f}")

# ---------- 4. RandomizedSearchCV ----------
print("\n=== STARTING RANDOM SEARCH ===")
start = time.time()

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=20,               # Try 20 random combinations
    cv=3,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
random_search.fit(X_train, y_train)

random_time = time.time() - start
print(f"RandomSearch completed in {random_time:.1f} seconds.")
print(f"Best Random Params: {random_search.best_params_}")
print(f"Best Random CV F1: {random_search.best_score_:.4f}")

# ---------- 5. Final Evaluation on the Test Set ----------
print("\n=== FINAL TEST SET EVALUATION ===")

# Baseline (default params from previous file)
baseline = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                         scale_pos_weight=2.76, eval_metric='logloss', random_state=42)
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)
print(f"Baseline Test F1: {f1_score(y_test, y_pred_base):.4f}")

# Best from GridSearch
best_grid = grid_search.best_estimator_
y_pred_grid = best_grid.predict(X_test)
print(f"Tuned Grid Test F1: {f1_score(y_test, y_pred_grid):.4f}")

# Best from RandomSearch
best_random = random_search.best_estimator_
y_pred_random = best_random.predict(X_test)
print(f"Tuned Random Test F1: {f1_score(y_test, y_pred_random):.4f}")

print("\n--- Tuned Random Search Classification Report ---")
print(classification_report(y_test, y_pred_random, target_names=['No Churn', 'Churn']))