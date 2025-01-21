import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Predicting house prices based on size (non-linear relationship)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])  # Quadratic relationship (y = x^2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Apply Ridge regression (regularization) to avoid overfitting
model = Ridge(alpha=1.0)  # Regularization parameter (higher alpha = more regularization)
model.fit(X_poly_train, y_train)

y_pred = model.predict(X_poly_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R²: {r2}")
print(f"MSE: {mse}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Plotting the graph
plt.scatter(X_test, y_test, color='blue', label='Actual Data')  
plt.scatter(X_test, y_pred, color='red', label='Predicted Data')  

# Create a smooth regression line by predicting over a range of values
X_range = np.linspace(min(X_test), max(X_test), 100).reshape(-1, 1)  # Create smooth range for X
X_range_poly = poly.transform(X_range)  # Transform the range using the polynomial features
y_range_pred = model.predict(X_range_poly)  # Get predictions for the smooth range

plt.plot(X_range, y_range_pred, color='green', label='Regression Line')  # Green curve for polynomial regression

plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression with Regularization (Ridge)')
plt.legend()
plt.show()