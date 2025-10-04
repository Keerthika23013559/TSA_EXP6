# Ex.No: 6       HOLT WINTERS METHOD

## NAME:KEETHIKA M P
## Register Number:212223240071
## Date: 04-10-2025

### AIM:
To implement the Holt Winters Method Model using Python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate the model  predictions against test data
6. Create teh final model and predict future data and plot it

### PROGRAM:

Importing necessary modules

```
# Importing necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
file_path = "/content/laptop_price.csv"   # Update path if needed
data = pd.read_csv(file_path, encoding="latin1")

# Sort by ID (to simulate time series order)
data = data.sort_values("laptop_ID")

# Use laptop_ID as pseudo-time index, Price as values
data_ts = pd.Series(data["Price_euros"].values,
                    index=pd.date_range(start="2018-01-01", periods=len(data), freq="MS"))

print("Laptop Price Time Series Head:")
print(data_ts.head())

# Plot original data
plt.figure(figsize=(12, 6))
data_ts.plot()
plt.title("Laptop Prices Over Time (Pseudo Series)")
plt.ylabel("Price (Euros)")
plt.xlabel("Date")
plt.grid(True)
plt.show()

# Resample (monthly)
data_monthly = data_ts.resample("MS").mean()

# Scale data
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(data_monthly.values.reshape(-1, 1))
scaled_data = pd.Series(scaled_values.flatten(), index=data_monthly.index)

plt.figure(figsize=(12, 6))
scaled_data.plot()
plt.title("Scaled Laptop Prices")
plt.ylabel("Scaled Price")
plt.xlabel("Date")
plt.grid(True)
plt.show()

# Seasonal decomposition (try with yearly seasonality = 12 months)
try:
    decomposition = seasonal_decompose(data_monthly, model='additive', period=12)
    decomposition.plot()
    plt.suptitle("Time Series Decomposition")
    plt.show()
except Exception as e:
    print(f"Decomposition error: {e}")

# Train-test split
scaled_data_adj = scaled_data + 0.1   # Ensure positive for multiplicative seasonality
train_size = int(len(scaled_data_adj) * 0.8)
train_data = scaled_data_adj[:train_size]
test_data = scaled_data_adj[train_size:]

print("Train size:", len(train_data))
print("Test size:", len(test_data))

# Holt-Winters model
model = ExponentialSmoothing(
    train_data,
    trend="add",
    seasonal="mul",
    seasonal_periods=12
).fit()

# Forecast
test_predictions = model.forecast(steps=len(test_data))

plt.figure(figsize=(12, 6))
train_data.plot(label="Training Data")
test_data.plot(label="Test Data")
test_predictions.plot(label="Forecast", linestyle="--")
plt.title("Holt-Winters Forecast (Laptop Prices)")
plt.ylabel("Scaled Price")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.show()

# Evaluation
rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
mae = mean_absolute_error(test_data, test_predictions)
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Final Model (on all data)
final_model = ExponentialSmoothing(
    scaled_data_adj,
    trend="add",
    seasonal="mul",
    seasonal_periods=12
).fit()

# Forecast next 12 months
future_steps = 12
future_predictions = final_model.forecast(steps=future_steps)
future_dates = pd.date_range(start=scaled_data_adj.index[-1] + pd.DateOffset(months=1),
                             periods=future_steps, freq="MS")
future_predictions.index = future_dates

# Convert back to original scale
original_data = pd.Series(
    scaler.inverse_transform(scaled_data_adj.values.reshape(-1, 1)).flatten(),
    index=scaled_data_adj.index
)

original_predictions = pd.Series(
    scaler.inverse_transform(future_predictions.values.reshape(-1, 1)).flatten(),
    index=future_predictions.index
)

# Plot final forecast
plt.figure(figsize=(14, 7))
original_data.plot(label="Historical Data", linewidth=2)
original_predictions.plot(label="Future Predictions", linestyle="--", linewidth=2)
plt.title("Laptop Price Forecast (Holt-Winters)")
plt.ylabel("Price (Euros)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.show()

print("\nFuture Predictions (in Euros):")
print(original_predictions)

```

### OUTPUT:

<img width="852" height="427" alt="image" src="https://github.com/user-attachments/assets/688088f5-acb7-4d4b-923d-e7a5f6d38cb5" />

<img width="816" height="438" alt="image" src="https://github.com/user-attachments/assets/cd795ee0-887a-435d-8505-493a32a9b51b" />

<img width="843" height="598" alt="image" src="https://github.com/user-attachments/assets/a3f58b40-8172-44d9-a0c6-c199edadf50f" />



<img width="845" height="433" alt="image" src="https://github.com/user-attachments/assets/0545bd23-fe11-4d28-a3cd-ba3cc2dd4651" />

<img width="819" height="423" alt="image" src="https://github.com/user-attachments/assets/9c02cda7-7d4b-4fba-89a1-4db173be24be" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
