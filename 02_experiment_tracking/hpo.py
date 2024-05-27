"""Random Forest Hyperparameter Optimization"""
import os
import pickle
import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("random-forest-hyperopt")


def load_pickle(filename: str):
    """Load a pickle file from the specified path.
    Args:
        filename (str): Path to the pickle file
    """
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int):
    """Optimize hyperparameters for a random forest model using Hyperopt.
    Args:
        data_path (str): Path to the directory containing the processed data
        num_trials (int): The number of parameter evaluations for the
            optimizer to explore
    """
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(params: dict):
        """Objective function for hyperopt optimization.
        Args:
            params (dict): Hyperparameters for the random forest model
        Returns:
            dict: Dictionary containing the loss value and status
        """
        with mlflow.start_run():
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform(
            'min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform(
            'min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    # for reproducible results
    rstate = np.random.default_rng(42)
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )


if __name__ == '__main__':
    run_optimization()
