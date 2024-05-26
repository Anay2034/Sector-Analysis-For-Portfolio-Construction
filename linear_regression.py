import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('Datafile.csv')

# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')

# Set the 'date' column as the index
data.set_index('date', inplace=True)

# Handle missing values (for simplicity, we use forward fill)
data.fillna(method='ffill', inplace=True)

# Select features (excluding 'Nifty_50' and any irrelevant columns)
features = ['GDP Billions of US $', 'Per Capita US $',
            'Growth Rate % Change', 'Change in growth Rate',
            'Growth Rate (%) in GDP per Capita', 'GNI Billions of US $',
            'Per Capita US $.1', 'GNI Growth Rate',
            'GNP Billions of US $', 'GNP Per Capita',
            'GNP Growth Rate', 'Inflation Rate (%)',
            'Inflation Rate Change', 'Output Manufacturing',
            'Manufacturing % of GDP']

# Create the feature matrix X and the target vector y
X = data[features]
y = data['Nifty_50']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Plot actual vs predicted against the date
plt.figure(figsize=(10, 6))
plt.plot(data.index[-len(y_test):], y_test, label='Actual')
plt.plot(data.index[-len(y_test):], y_pred, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Nifty_50')
plt.title('Actual vs Predicted Nifty_50')
plt.legend()
plt.show()
