import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
train = pd.read_csv("fashion-mnist_train.csv")
test = pd.read_csv("fashion-mnist_test.csv")

# Separate features and labels
X_train = train.iloc[:, 1:].values / 255.0
y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1:].values / 255.0
y_test = test.iloc[:, 0].values

# Try different K values
for k in [1, 3, 5]:
    print("\nK =", k)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train[:10000], y_train[:10000])

    y_pred = knn.predict(X_test[:2000])

    print("Accuracy:", accuracy_score(y_test[:2000], y_pred))
    print(classification_report(y_test[:2000], y_pred, zero_division=0))
