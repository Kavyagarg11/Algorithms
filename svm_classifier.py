from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic classification data
# Adjusting parameters to avoid the error by ensuring that the sum of informative, redundant, and repeated features is less than the total number of features
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM Classifier with a linear kernel
svm_classifier = SVC(kernel='linear', random_state=42)

svm_classifier.fit(X_train, y_train)

print(f"Model Accuracy: {svm_classifier.score(X_test, y_test):.2f}")

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Display the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Plot decision boundary
plt.figure(figsize=(8, 6))
ax = plt.gca()

# Create grid points to plot the decision boundary
xx, yy = np.meshgrid(np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100),
                     np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100))

# Predict for each point in the grid
Z = grid_search.best_estimator_.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
ax.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')

# Plot the training points
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o', cmap='coolwarm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Classifier with Decision Boundary')
plt.legend(*scatter.legend_elements(), title="Classes")
plt.show()