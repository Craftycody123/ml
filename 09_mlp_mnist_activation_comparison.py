
import matplotlib.pyplot as plt
import time
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.utils import to_categorical

# Load downloaded dataset
train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")

X_train = train.iloc[:, 1:].values.reshape(-1, 28, 28)
y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1:].values.reshape(-1, 28, 28)
y_test = test.iloc[:, 0].values

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def create_model(activation):
    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),
        Dense(128, activation=activation),
        Dense(64, activation=activation),
        Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

activations = ["sigmoid", "tanh", "relu"]
results = {}
histories = {}

for act in activations:
    print(f"\nTraining with {act} activation")
    model = create_model(act)

    start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.2,
        verbose=1
    )
    end = time.time()

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    results[act] = {"accuracy": acc, "time": end - start}
    histories[act] = history

for act in results:
    print(
        f"{act} -> Accuracy: {results[act]['accuracy']:.4f}, "
        f"Time: {results[act]['time']:.2f}s"
    )

for act in activations:
    plt.plot(histories[act].history["val_accuracy"], label=act)

plt.title("Validation Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

for act in activations:
    plt.plot(histories[act].history["val_loss"], label=act)

plt.title("Validation Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
