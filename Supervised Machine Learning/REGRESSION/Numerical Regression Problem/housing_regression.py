import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ---------- 1. Load Data ----------
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()

X = df.drop(columns=['MedHouseVal']) # Features
y = df['MedHouseVal']                # Target (House Price in $100,000s)

print(f"Dataset Shape: {X.shape}")
print(f"Target Range: {y.min():.2f} to {y.max():.2f} (in $100,000s)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- 2. Train Models ----------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Regression Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    print(f"\n--- {name} ---")
    print(f"RMSE: ${rmse * 100000:,.0f}")
    print(f"MAE:  ${mae * 100000:,.0f}")
    print(f"R²:   {r2:.4f}")

# ---------- 3. Visualize Predictions (The Regression "Confusion Matrix") ----------

best_model = models["XGBoost"]
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.3, color='teal')

# Draw the "Perfect Prediction" line (y = x)
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

plt.xlabel("Actual Price ($100k)")
plt.ylabel("Predicted Price ($100k)")
plt.title("XGBoost: Actual vs Predicted House Prices")
plt.legend()
plt.grid(alpha=0.3)
plt.show()