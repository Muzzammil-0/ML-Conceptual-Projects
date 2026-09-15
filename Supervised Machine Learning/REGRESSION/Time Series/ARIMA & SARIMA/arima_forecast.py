import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# ---------- 1. Load Data (Same as before) ----------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url)
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Chronological Split (Same 80/20 split)
train_size = int(len(df) * 0.8)
train_data = df['Passengers'].iloc[:train_size]
test_data = df['Passengers'].iloc[train_size:]

# ---------- 2. The ARIMA Model ----------
# ARIMA(p, d, q):
# p = AutoRegressive (lags of the series itself)
# d = Integrated (differencing to remove trend)
# q = Moving Average (lags of the forecast errors)
# (1,1,1) is a standard starting point for simple data.
print("Fitting ARIMA model...")
model = ARIMA(train_data, order=(1, 1, 1))
fitted_model = model.fit()

# Forecast the future
forecast = fitted_model.forecast(steps=len(test_data))

# ---------- 3. Evaluate ----------
rmse = np.sqrt(mean_squared_error(test_data, forecast))
mape = mean_absolute_percentage_error(test_data, forecast) * 100

print(f"\nARIMA RMSE: {rmse:.2f}")
print(f"ARIMA MAPE: {mape:.2f}%")

# ---------- 4. Plot ----------
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data, label='Train Data', color='blue')
plt.plot(test_data.index, test_data, label='Actual Future', color='green')
plt.plot(test_data.index, forecast, label='ARIMA Forecast', color='purple', linestyle='--')

plt.title('Airline Passenger Forecast (ARIMA)')
plt.xlabel('Year')
plt.ylabel('Passengers')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
