import argparse

import matplotlib as mpl
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from auto_ml_flow import AutoMLFlow

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

    AutoMLFlow.set_tracking_url("http://87.242.117.47:8811")
    
    with AutoMLFlow.experiment_manager("My experiment for test") as experiment:
        with AutoMLFlow.run_manager(experiment, description=f"Some shit happens here???"):
            # Train model and track evaluation metrics
            params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "learning_rate": args.learning_rate,
                "eval_metric": "mlogloss",
                "colsample_bytree": args.colsample_bytree,
                "subsample": args.subsample,
                "seed": 42,
            }
            eval_results = {} # Dictionary to store evaluation results
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
                AutoMLFlow.log_metric("log_loss", float(loss))

            # Predict on the test set
            y_proba = model.predict(dtest)
            y_pred = y_proba.argmax(axis=1)
            loss = log_loss(y_test, y_proba)
            acc = accuracy_score(y_test, y_pred)

            # Add final evaluation metrics to the run
            AutoMLFlow.log_metric("log_loss", loss)
            AutoMLFlow.log_metric("accuracy", acc)


if __name__ == "__main__":
    main()
