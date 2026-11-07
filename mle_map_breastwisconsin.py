import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = load_breast_cancer()
X = data.data
y = data.target

# -------------------------------
# 2. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 3. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 4. Train Models
# -------------------------------

# MLE (No Regularization)
mle_model = LogisticRegression(penalty=None, max_iter=5000)
mle_model.fit(X_train, y_train)

# MAP - L2 Regularization
l2_model = LogisticRegression(penalty='l2', C=1.0, max_iter=5000)
l2_model.fit(X_train, y_train)

# MAP - L1 Regularization
l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
l1_model.fit(X_train, y_train)

# -------------------------------
# 5. Predictions
# -------------------------------
y_pred_mle = mle_model.predict(X_test)
y_pred_l2 = l2_model.predict(X_test)
y_pred_l1 = l1_model.predict(X_test)

# -------------------------------
# 6. Accuracy Comparison
# -------------------------------
print("\n=== Accuracy ===")
print("MLE Accuracy:", accuracy_score(y_test, y_pred_mle))
print("L2 MAP Accuracy:", accuracy_score(y_test, y_pred_l2))
print("L1 MAP Accuracy:", accuracy_score(y_test, y_pred_l1))

# -------------------------------
# 7. Classification Reports
# -------------------------------
print("\n=== Classification Report: MLE ===")
print(classification_report(y_test, y_pred_mle))

print("\n=== Classification Report: L2 MAP ===")
print(classification_report(y_test, y_pred_l2))

print("\n=== Classification Report: L1 MAP ===")
print(classification_report(y_test, y_pred_l1))

# -------------------------------
# 8. Coefficient Comparison
# -------------------------------
coeffs = pd.DataFrame({
    "Feature": data.feature_names,
    "MLE": mle_model.coef_[0],
    "L2": l2_model.coef_[0],
    "L1": l1_model.coef_[0]
})

print("\n=== Sample Coefficients ===")
print(coeffs.head(10))

# -------------------------------
# 9. Visualization
# -------------------------------
plt.figure()

plt.plot(np.abs(mle_model.coef_[0]), label="MLE")
plt.plot(np.abs(l2_model.coef_[0]), label="L2 (MAP)")
plt.plot(np.abs(l1_model.coef_[0]), label="L1 (MAP)")

plt.xlabel("Feature Index")
plt.ylabel("Coefficient Magnitude")
plt.title("Coefficient Comparison: MLE vs MAP")
plt.legend()
plt.grid()

plt.show()

# -------------------------------
# 10. Sparsity Check (L1)
# -------------------------------
num_zero = np.sum(l1_model.coef_[0] == 0)
print("\nNumber of zero coefficients in L1 (feature selection):", num_zero)