```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Bootstrapping
acc, f1 = [], []

np.random.seed(42)

for i in range(10):
    indices = np.random.choice(len(X), len(X), replace=True)

    X_train = X[indices]
    y_train = y[indices]

    # Out-of-bag samples
    oob = np.setdiff1d(np.arange(len(X)), np.unique(indices))

    if len(oob) == 0:
        continue

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X[oob])

    acc.append(accuracy_score(y[oob], pred))
    f1.append(f1_score(y[oob], pred, average='weighted'))

print("Bootstrapping")
print("Accuracy:", np.mean(acc))
print("F1 Score:", np.mean(f1))

# 3. K-Fold Cross Validation
model = DecisionTreeClassifier(random_state=42)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
f1 = cross_val_score(model, X, y, cv=kfold, scoring='f1_weighted')

print("\nK-Fold Cross Validation")
print("Accuracy:", np.mean(acc))
print("F1 Score:", np.mean(f1))
```