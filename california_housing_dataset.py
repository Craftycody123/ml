import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = fetch_california_housing()

X = data.data[:, 2].reshape(-1, 1)  # AveRooms
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize input
mean = X_train.mean()
std = X_train.std()

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Add bias column
X_train = np.c_[np.ones(len(X_train)), X_train]
X_test = np.c_[np.ones(len(X_test)), X_test]

# Gradient Descent
theta = np.zeros(2)
lr = 0.01

for i in range(1000):
    y_pred = X_train @ theta
    error = y_pred - y_train
    gradient = (X_train.T @ error) / len(y_train)
    theta -= lr * gradient

pred_gd = X_test @ theta

# Normal Equation
theta_ne = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
pred_ne = X_test @ theta_ne

# Evaluation
print("Gradient Descent")
print("MSE:", mean_squared_error(y_test, pred_gd))
print("R2:", r2_score(y_test, pred_gd))

print("\nNormal Equation")
print("MSE:", mean_squared_error(y_test, pred_ne))
print("R2:", r2_score(y_test, pred_ne))

# Visualization
plt.scatter(X_test[:, 1], y_test, label="Actual")
plt.plot(X_test[:, 1], pred_gd, color="red", label="Predicted")
plt.xlabel("Average Rooms (Normalized)")
plt.ylabel("House Value")
plt.title("Linear Regression")
plt.legend()
plt.show()
