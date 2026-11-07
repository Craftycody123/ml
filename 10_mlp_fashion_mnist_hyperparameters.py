
import time
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Load downloaded dataset
train = pd.read_csv("fashion-mnist_train.csv")
test = pd.read_csv("fashion-mnist_test.csv")

X_train = train.iloc[:, 1:].values.reshape(-1, 28, 28)
y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1:].values.reshape(-1, 28, 28)
y_test = test.iloc[:, 0].values

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def create_model(learning_rate):
    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

learning_rates = [0.001, 0.01]
batch_sizes = [32, 128]
epochs_list = [5, 10]
results = []

for lr in learning_rates:
    for batch in batch_sizes:
        for epochs in epochs_list:
            print(f"\nLR={lr}, Batch={batch}, Epochs={epochs}")
            model = create_model(lr)

            start = time.time()
            model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch,
                validation_split=0.2,
                verbose=0
            )
            end = time.time()

            loss, acc = model.evaluate(X_test, y_test, verbose=0)

            results.append({
                "lr": lr,
                "batch": batch,
                "epochs": epochs,
                "accuracy": acc,
                "time": end-start
            })

            print(f"Accuracy: {acc:.4f}, Time: {end-start:.2f}s")

for result in results:
    print(result)
