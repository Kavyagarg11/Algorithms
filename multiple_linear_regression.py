import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = {
    'Size': [1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500, 4000, 4200],
    'Bedrooms': [3, 4, 3, 4, 5, 5, 6, 6, 7, 8],
    'Age': [10, 5, 7, 3, 12, 8, 10, 15, 20, 5],
    'Price': [400000, 450000, 500000, 600000, 700000, 750000, 800000, 850000, 900000, 1000000]
}

df = pd.DataFrame(data)

X = df[['Size', 'Bedrooms', 'Age']]  
y = df['Price']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)  # R² is used to evaluate how well the model fits the data
mse = mean_squared_error(y_test, y_pred)  # MSE is the mean squared error between predictions and actual values

# Print results
print(f"R²: {r2}")
print(f"MSE: {mse}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Plotting the Actual vs Predicted prices for the test set (using Size as the x-axis)
plt.scatter(X_test['Size'], y_test, color='blue', label='Actual Price')
plt.scatter(X_test['Size'], y_pred, color='red', label='Predicted Price')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price')
plt.title('Multiple Linear Regression: Actual vs Predicted')
plt.legend()
plt.show()