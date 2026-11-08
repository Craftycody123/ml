import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load Fashion MNIST dataset
data = fetch_openml(data_id=40996, as_frame=False)

X = data.data[:5000] / 255.0
y = data.target[:5000].astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameters
learning_rates = [0.001, 0.01]
batch_sizes = [32, 64]
epochs = [10, 20]

# Try different combinations
for lr in learning_rates:
    for batch in batch_sizes:
        for ep in epochs:

            model = MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                solver="adam",
                learning_rate_init=lr,
                batch_size=batch,
                max_iter=ep,
                random_state=42
            )

            start = time.time()
            model.fit(X_train, y_train)
            end = time.time()

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print("Learning Rate:", lr)
            print("Batch Size:", batch)
            print("Epochs:", ep)
            print("Accuracy:", round(accuracy * 100, 2), "%")
            print("Training Time:", round(end - start, 2), "seconds")
            print("-------------------------")