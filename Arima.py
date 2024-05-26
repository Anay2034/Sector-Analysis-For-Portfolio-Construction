import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import warnings
import itertools

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv('Datafile.csv')

# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')

# Set the 'date' column as the index
data.set_index('date', inplace=True)

# Handle missing values (for simplicity, we use forward fill)
data.fillna(method='ffill', inplace=True)

# Extract the Nifty 50 index time series
nifty_series = data['Nifty_50']

# Split the data into train and test sets
train_size = int(len(nifty_series) * 0.8)
train, test = nifty_series[:train_size], nifty_series[train_size:]

# Function to find the best ARIMA parameters
def find_best_arima(train):
    p = d = q = range(0, 6)
    pdq = list(itertools.product(p, d, q))

    best_aic = float("inf")
    best_pdq = None

    for param in pdq:
        try:
            temp_model = ARIMA(train, order=param)
            temp_model_fit = temp_model.fit()
            if temp_model_fit.aic < best_aic:
                best_aic = temp_model_fit.aic
                best_pdq = param
        except:
            continue

    return best_pdq, best_aic

# Find the best ARIMA parameters
best_pdq, best_aic = find_best_arima(train)
print(f'Best ARIMA parameters: {best_pdq} with AIC: {best_aic}')

# Fit the ARIMA model with the best parameters
best_arima_model = ARIMA(train, order=best_pdq)
best_arima_fit = best_arima_model.fit()

# Make predictions with the best model
start_index = test.index[0]
end_index = test.index[-1]
arima_predictions_best = best_arima_fit.predict(start=start_index, end=end_index, typ='levels')

# Evaluate the model with best parameters
mse_best = mean_squared_error(test, arima_predictions_best)
r2_best = r2_score(test, arima_predictions_best)

print(f'Best Model Mean Squared Error: {mse_best}')
print(f'Best Model R^2 Score: {r2_best}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Actual Data', color='blue')
plt.plot(arima_predictions_best.index, arima_predictions_best, label='Predicted Data', color='red')
plt.title('Nifty 50 Index Prediction using ARIMA')
plt.xlabel('Date')
plt.ylabel('Nifty 50 Index')
plt.legend()
plt.show()

# Function to fit ARIMA model and calculate MAE
def fit_arima_and_calculate_mae(train, test, order):
    arima_model = ARIMA(train, order=order)
    arima_fit = arima_model.fit()
    arima_predictions = arima_fit.predict(start=test.index[0], end=test.index[-1], typ='levels')
    mae = np.mean(np.abs(test - arima_predictions))
    return arima_fit, mae

# Find the best ARIMA model based on MAE
best_mae = float("inf")
best_arima_fit = None
best_order = None

for param in pdq:
    try:
        _, mae = fit_arima_and_calculate_mae(train, test, param)
        if mae < best_mae:
            best_mae = mae
            best_order = param
            best_arima_fit, _ = fit_arima_and_calculate_mae(train, test, param)
    except:
        continue

# Make predictions with the best model
arima_predictions_best_mae = best_arima_fit.predict(start=test.index[0], end=test.index[-1], typ='levels')

# Evaluate the best model
mse_best_mae = mean_squared_error(test, arima_predictions_best_mae)
r2_best_mae = r2_score(test, arima_predictions_best_mae)

print(f'Best Model Mean Squared Error using MAE: {mse_best_mae}')
print(f'Best Model R^2 Score using MAE: {r2_best_mae}')
print(f'Best ARIMA parameters using MAE: {best_order}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Actual Data', color='blue')
plt.plot(arima_predictions_best_mae.index, arima_predictions_best_mae, label='Predicted Data', color='red')
plt.title('Nifty 50 Index Prediction using ARIMA (Best Model by MAE)')
plt.xlabel('Date')
plt.ylabel('Nifty 50 Index')
plt.legend()
plt.show()
