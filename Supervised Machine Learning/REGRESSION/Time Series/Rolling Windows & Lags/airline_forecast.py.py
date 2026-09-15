import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# ---------- 1. Load Data ----------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url)

# Convert Month to datetime and set as index
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

print(f"Dataset shape: {df.shape}")
print(df.head())

# ---------- 2. Feature Engineering (The Time Series Magic) ----------
# We create "Lag" features. The model needs yesterday's data to predict today's.
# shift(1) moves the data down by 1 row.
df['lag_1'] = df['Passengers'].shift(1)
df['lag_12'] = df['Passengers'].shift(12) # 12 months ago (captures yearly seasonality)
df['rolling_mean_3'] = df['Passengers'].shift(1).rolling(window=3).mean()

# Extract month number (1-12) to help the model learn seasonality
df['month'] = df.index.month

# Drop the first 12 rows because they now contain NaN (due to the shifts)
df = df.dropna()

# ---------- 3. Chronological Train/Test Split ----------
# We cannot use train_test_split(random_state=42). 
# We must take the first 80% of time as train, and the last 20% as test.
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

features = ['lag_1', 'lag_12', 'rolling_mean_3', 'month']
X_train, y_train = train[features], train['Passengers']
X_test, y_test = test[features], test['Passengers']

print(f"\nTraining on {len(train)} months (Past)")
print(f"Testing on {len(test)} months (Future)")

# ---------- 4. Train XGBoost ----------
model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# ---------- 5. Evaluate ----------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print(f"\nRMSE: {rmse:.2f} passengers")
print(f"MAPE: {mape:.2f}% (average error percentage)")

# ---------- 6. Plot the Forecast ----------
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Passengers'], label='Train Data (Historical)', color='blue')
plt.plot(test.index, y_test, label='Test Data (Actual Future)', color='green')
plt.plot(test.index, y_pred, label='XGBoost Forecast', color='red', linestyle='--')

plt.title('Airline Passenger Forecast (Time Series)')
plt.xlabel('Year')
plt.ylabel('Passengers')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Look at a few predictions
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.round(0)})
print("\nActual vs Predicted (First 5 test months):")
print(comparison.head())