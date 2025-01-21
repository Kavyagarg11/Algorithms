import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Regressor with pruning (max_depth, min_samples_leaf)
regressor = DecisionTreeRegressor(criterion='squared_error', random_state=42, max_depth=5, min_samples_leaf=5)

regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = regressor.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plotting the results
plt.figure(figsize=(8, 6))

# Plot the training data
plt.scatter(X_train, y_train, color='blue', label='Training data')

# Plot the test data
plt.scatter(X_test, y_test, color='green', label='Test data')

# Generate predictions using the trained model on the training data points
y_train_pred = regressor.predict(X_train)

# Sort training data for correct plotting
X_train_sorted = np.sort(X_train, axis=0)

# Plot the step curve using the predictions for the sorted training data
plt.step(X_train_sorted.flatten(), y_train_pred[np.argsort(X_train.flatten())], color='red', label='Prediction (Step Curve)', linewidth=2)

# Adding labels and legend
plt.title('Decision Tree Regression (Pruned Tree with Recursive Splits)')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()

plt.show()

# Plot the Decision Tree 
plt.figure(figsize=(12, 8))
plot_tree(regressor, filled=True, feature_names=["Feature"], fontsize=12)
plt.title("Visualizing the Decision Tree (Pruned)")
plt.show()