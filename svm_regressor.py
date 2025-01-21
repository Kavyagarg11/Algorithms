from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the SVM Regressor with RBF kernel
svm_regressor = SVR(kernel='rbf')

svm_regressor.fit(X_train, y_train)

y_pred = svm_regressor.predict(X_test)

print(f"R-squared Score: {svm_regressor.score(X_test, y_test)}")

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVR(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

# Plotting the results
plt.scatter(X_test, y_test, color='red', label="True values")
plt.scatter(X_test, y_pred, color='blue', label="Predictions")
plt.title('SVM Regressor Results')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
