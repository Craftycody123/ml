import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Load Dataset
# -----------------------------
data = load_diabetes()
print(data.DESCR)
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# -----------------------------
# Feature and Target
# -----------------------------
X = df.drop('target', axis=1)
y = df['target']

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Linear Regression
# -----------------------------
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

# -----------------------------
# Ridge Regression
# -----------------------------
ridge_alphas = np.logspace(-3, 3, 50)
ridge_model = RidgeCV(alphas=ridge_alphas, cv=5)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# -----------------------------
# Lasso Regression
# -----------------------------
lasso_model = LassoCV(cv=5, max_iter=10000)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
models = {
    "Linear": y_pred_lin,
    "Ridge": y_pred_ridge,
    "Lasso": y_pred_lasso
}

print("\nModel Performance:")
print("----------------------------------")
print("Model\t\tMSE\t\tR²")
print("----------------------------------")

for name, y_pred in models.items():
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}\t\t{mse:.4f}\t{r2:.4f}")

# -----------------------------
# Graph 1: Actual vs Predicted
# -----------------------------
plt.figure()

plt.scatter(y_test, y_pred_lin, label="Linear")
plt.scatter(y_test, y_pred_ridge, label="Ridge")
plt.scatter(y_test, y_pred_lasso, label="Lasso")

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Comparison")
plt.legend()

plt.show()

# -----------------------------
# Graph 2: Coefficient Comparison
# -----------------------------
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Linear': lin_model.coef_,
    'Ridge': ridge_model.coef_,
    'Lasso': lasso_model.coef_
})


print("\nCoefficients:")
print(coef_df)
coef_df.set_index('Feature').plot(kind='bar')

plt.title("Coefficient Comparison (Shrinkage Effect)")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)

plt.show()

# -----------------------------
# Best Alpha Values
# -----------------------------
print("\nBest Alpha Values:")
print("Ridge Alpha:", ridge_model.alpha_)
print("Lasso Alpha:", lasso_model.alpha_)
