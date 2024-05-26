import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
features = [
    ' GDP  Billions of US $', ' Per Capita US $', ' Growth Rate % Change', ' Change in growth Rate',
    'Growth Rate (%) in GDP per Capita', ' GNI Billions of US $', ' Per Capita US $.1', 'GNI  Growth Rate  ',
    ' GNP Billions of US $', 'GNP  Per Capita', 'GNP  Growth Rate', ' Inflation Rate (%)', ' Inflation Rate Change',
    '  Output Manufacturing', 'Manufacturing % of GDP'
]

# Create the feature matrix X and the target vector y
X = data[features]
y = data['Nifty_50']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to evaluate and print model performance
def evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    return model

# Evaluate Random Forest Regressor
print("Random Forest Regressor:")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
evaluate_model(rf_model, X_train_scaled, y_train, X_test_scaled, y_test)

# Evaluate Gradient Boosting Regressor
print("\nGradient Boosting Regressor:")
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
evaluate_model(gb_model, X_train_scaled, y_train, X_test_scaled, y_test)

# Cross-Validation for Random Forest
print("\nCross-Validation for Random Forest:")
tscv = TimeSeriesSplit(n_splits=5)
rf_cv_scores = cross_val_score(rf_model, scaler.transform(X), y, cv=tscv, scoring='r2')
print(f'Cross-Validation R^2 Scores: {rf_cv_scores}')
print(f'Average Cross-Validation R^2 Score: {np.mean(rf_cv_scores)}')

# Cross-Validation for Gradient Boosting
print("\nCross-Validation for Gradient Boosting:")
gb_cv_scores = cross_val_score(gb_model, scaler.transform(X), y, cv=tscv, scoring='r2')
print(f'Cross-Validation R^2 Scores: {gb_cv_scores}')
print(f'Average Cross-Validation R^2 Score: {np.mean(gb_cv_scores)}')

# Feature Importance for Random Forest
print("\nFeature Importance (Random Forest):")
rf_feature_importance = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print(rf_feature_importance)

# Feature Importance for Gradient Boosting
print("\nFeature Importance (Gradient Boosting):")
gb_feature_importance = pd.Series(gb_model.feature_importances_, index=features).sort_values(ascending=False)
print(gb_feature_importance)
