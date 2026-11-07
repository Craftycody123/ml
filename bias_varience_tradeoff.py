
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 1. Load Boston Housing dataset
url = "http://lib.stat.cmu.edu/datasets/boston"
raw = pd.read_csv(url, sep=r"\s+", skiprows=22, header=None)

X = np.hstack([raw.values[::2, :], raw.values[1::2, :2]])
y = raw.values[1::2, 2]

# Use RM (average number of rooms)
X = X[:, 5].reshape(-1, 1)

# 2. Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Polynomial Regression for different degrees
degrees = [1, 2, 3, 5, 8]
train_errors = []
val_errors = []

for d in degrees:
    model = make_pipeline(
        PolynomialFeatures(degree=d),
        StandardScaler(),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_errors.append(mean_squared_error(y_train, train_pred))
    val_errors.append(mean_squared_error(y_val, val_pred))

    print("Degree:", d)
    print("Training Error:", train_errors[-1])
    print("Validation Error:", val_errors[-1])
    print()

# 4. Plot errors
plt.plot(degrees, train_errors, marker='o', label='Training Error')
plt.plot(degrees, val_errors, marker='o', label='Validation Error')

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Bias-Variance Tradeoff")
plt.legend()
plt.show()
