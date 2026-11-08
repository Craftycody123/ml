import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load MNIST dataset
mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data / 255.0
y = mnist.target.astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Activation functions
activations = ["logistic", "relu", "tanh"]

for act in activations:

    # Create model
    model = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation=act,
        solver="adam",
        max_iter=5,
        random_state=42
    )

    # Train model and calculate time
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    # Test model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(act.upper())
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Training Time:", round(end - start, 2), "seconds")
    print("-------------------------")