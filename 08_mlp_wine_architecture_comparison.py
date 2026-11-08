# EXP 13 - MLP Wine Quality

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("winequality-red.csv")

X = df.drop("quality", axis=1)
y = df["quality"]

# Convert quality into classification labels
# (quality values remain separate classes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

architectures = [
    (50,),
    (100,),
    (100, 50),
    (100, 50, 25)
]

for arch in architectures:
    model = MLPClassifier(
        hidden_layer_sizes=arch,
        activation="relu",
        max_iter=500,
        random_state=42
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\nArchitecture:", arch)
    print("Accuracy:", accuracy_score(y_test, pred))