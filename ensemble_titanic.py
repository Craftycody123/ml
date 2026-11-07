import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load dataset
data = pd.read_csv("train.csv")

X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']

# 2. Preprocessing
num_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
cat_cols = ['Sex', 'Embarked']

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Bagging
bagging = Pipeline([
    ('preprocess', preprocessor),
    ('model', BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        random_state=42
    ))
])

bagging.fit(X_train, y_train)
bag_pred = bagging.predict(X_test)

# 5. Boosting (AdaBoost)
boosting = Pipeline([
    ('preprocess', preprocessor),
    ('model', AdaBoostClassifier(
        n_estimators=50,
        random_state=42
    ))
])

boosting.fit(X_train, y_train)
boost_pred = boosting.predict(X_test)

# 6. Evaluation
def evaluate(name, y_test, pred):
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall:", recall_score(y_test, pred))
    print("F1 Score:", f1_score(y_test, pred))

evaluate("Bagging", y_test, bag_pred)
evaluate("Boosting", y_test, boost_pred)
