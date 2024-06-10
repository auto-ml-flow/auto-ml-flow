import argparse
import time

import matplotlib as mpl
import numpy as np
from sklearn.metrics import log_loss
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from auto_ml_flow import AutoMLFlow

mpl.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="Keras example")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="learning rate for the optimizer (default: 0.001)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="batch size for training (default: 32)"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="number of epochs for training (default: 5)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Preprocess data
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    AutoMLFlow.set_tracking_url("https://87.242.117.47/")
    AutoMLFlow.start_experiment("MNIST Classification")

    with AutoMLFlow.start_run("Keras_MNIST"):
        # Log hyperparameters
        AutoMLFlow.log_metric("learning_rate", args.learning_rate)
        AutoMLFlow.log_metric("batch_size", args.batch_size)
        AutoMLFlow.log_metric("epochs", args.epochs)

        # Define model
        model = Sequential(
            [
                Flatten(input_shape=(28 * 28,)),
                Dense(256, activation="relu"),
                Dense(10, activation="softmax"),
            ]
        )

        # Compile model
        optimizer = Adam(learning_rate=args.learning_rate)
        loss_fn = SparseCategoricalCrossentropy(from_logits=False)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
        # Train model
        history = model.fit(
            X_train,
            y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(X_test, y_test),
        )

        # Log training history
        for metric in history.history:
            for value in history.history[metric]:
                AutoMLFlow.log_metric(metric, value)

        # Evaluate model
        loss, acc = model.evaluate(X_test, y_test)
        AutoMLFlow.log_metric("test_accuracy", acc)
        AutoMLFlow.log_metric("test_loss", loss)

        # Log evaluation results
        y_proba = model.predict(X_test)
        y_pred = np.argmax(y_proba, axis=1)
        log_loss_value = log_loss(y_test, y_proba)
        AutoMLFlow.log_metric("log_loss", log_loss_value)

        # Log dataset information
        n_samples, n_features = X_train.shape
        AutoMLFlow.log_dataset(n_features, n_samples, X_train)

        # Print prediction time
        start_time = time.time()
        model.predict(X_test[:1])  # Make a prediction on the first 10 samples
        end_time = time.time()
        prediction_time = end_time - start_time
        print(f"Prediction time for 10 samples: {prediction_time:.5f} seconds")


if __name__ == "__main__":
    main()
