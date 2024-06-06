import argparse
from time import sleep

import matplotlib as mpl
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from auto_ml_flow import AutoMLFlow

mpl.use("Agg")

description = """
This script trains an XGBoost model on the Iris dataset and
logs various parameters, metrics, and results using the AutoMLFlow library. 
The script uses command-line arguments to set the model's hyperparameters, 
making it flexible and easy to experiment with different configurations. 
It also integrates AutoMLFlow for tracking and managing machine learning experiments. 
This includes logging the dataset details, model parameters, 
and evaluation metrics at each step of the training process. 
"""

def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost example")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate to update step size at each boosting step (default: 0.3)",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.1,
        help="Subsample ratio of columns when constructing each tree (default: 1.0)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.1,
        help="Subsample ratio of the training instances (default: 1.0)",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the Iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set up AutoMLFlow for tracking the experiment
    AutoMLFlow.set_tracking_url("http://localhost:8000/")
    AutoMLFlow.start_experiment(name="Dataset maining", description=description)
    
    with AutoMLFlow.start_run("Test123"):
        n_samples, n_features = X.shape
        AutoMLFlow.log_dataset(n_features, n_samples, X)
        AutoMLFlow.predict_training_time()
        # Define model parameters
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "learning_rate": args.learning_rate,
            "eval_metric": "mlogloss",
            "colsample_bytree": args.colsample_bytree,
            "subsample": args.subsample,
            "seed": 42,
        }
        
        # Log model parameters
        for key, value in params.items():
            AutoMLFlow.log_param(key, value)
            
        # Train the model and store evaluation results
        eval_results = {}
        model = xgb.train(
            params,
            dtrain,
            evals=[(dtrain, "train"), (dtest, "test")],
            evals_result=eval_results,
        )

        # Log evaluation metrics at each step
        for loss in eval_results["test"]["mlogloss"]:
            AutoMLFlow.log_metric("log_loss", float(loss))

        # Make predictions and evaluate the model
        y_proba = model.predict(dtest)
        y_pred = y_proba.argmax(axis=1)
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        # Log final evaluation metrics and dataset details
        AutoMLFlow.log_result("log_loss", loss)
        AutoMLFlow.log_result("accuracy", acc)

if __name__ == "__main__":
    main()
