import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load dataset
train = pd.read_csv("fashion-mnist_train.csv")
test = pd.read_csv("fashion-mnist_test.csv")

# Separate features and labels
X_train = train.iloc[:5000, 1:].values / 255.0
y_train = train.iloc[:5000, 0].values

X_test = test.iloc[:1000, 1:].values / 255.0
y_test = test.iloc[:1000, 0].values

# Linear SVM
linear_model = svm.SVC(kernel="linear")
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Polynomial SVM
poly_model = svm.SVC(kernel="poly", degree=3)
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)

# RBF SVM
rbf_model = svm.SVC(kernel="rbf")
rbf_model.fit(X_train, y_train)
y_pred_rbf = rbf_model.predict(X_test)

# Compare accuracies
print("Linear SVM:", accuracy_score(y_test, y_pred_linear))
print("Polynomial SVM:", accuracy_score(y_test, y_pred_poly))
print("RBF SVM:", accuracy_score(y_test, y_pred_rbf))
