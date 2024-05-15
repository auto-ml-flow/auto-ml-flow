import argparse

import matplotlib as mpl
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from auto_ml_flow.client import v1
from auto_ml_flow.experiments import experiment_manager
from auto_ml_flow.run_metrics import add_metric_to
from auto_ml_flow.runs import run_manager

mpl.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost example")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.3,
        help="learning rate to update step size at each boosting step (default: 0.3)",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=1.0,
        help="subsample ratio of columns when constructing each tree (default: 1.0)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="subsample ratio of the training instances (default: 1.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    client = v1.AutoMLFlowClient(base_url="http://87.242.117.47:8811") # public deployment

    with experiment_manager(client, "My experiment for test") as experiment:
        for seed in [1, 2, 34, 42]:
            with run_manager(client, experiment, description=f"Run with custom seed {seed}") as run:
                # Train model and track evaluation metrics
                params = {
                    "objective": "multi:softprob",
                    "num_class": 3,
                    "learning_rate": args.learning_rate,
                    "eval_metric": "mlogloss",
                    "colsample_bytree": args.colsample_bytree,
                    "subsample": args.subsample,
                    "seed": seed,
                }
                eval_results = {}  # Dictionary to store evaluation results
                model = xgb.train(
                    params,
                    dtrain,
                    evals=[(dtrain, "train"), (dtest, "test")],
                    evals_result=eval_results,  # Store evaluation results during training
                )

                # Extract log loss values at each step from eval_results
                log_loss_values = eval_results["test"]["mlogloss"]

                # Log log loss at each step
                for loss in log_loss_values:
                    add_metric_to(run, "log_loss", float(loss), client)

                # Predict on the test set
                y_proba = model.predict(dtest)
                y_pred = y_proba.argmax(axis=1)
                loss = log_loss(y_test, y_proba)
                acc = accuracy_score(y_test, y_pred)

                # Add final evaluation metrics to the run
                add_metric_to(run, "seed", float(seed), client)
                add_metric_to(run, "log_loss", float(loss), client)
                add_metric_to(run, "accuracy", float(acc), client)


if __name__ == "__main__":
    main()
