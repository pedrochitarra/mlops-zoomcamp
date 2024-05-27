"""Train a random forest model and log the RMSE metric."""
import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow

# Q1. Install MLflow
# What's the version that you have?
# 2.13.0

# Q2. Download and preprocess the data
# How many files were saved to OUTPUT_FOLDER?
# 4

# Q3. Train a model with autolog
# What is the value of the min_samples_split parameter?
# 2

# Q4. Launch the tracking server locally
# In addition to backend-store-uri, what else do you need to pass to
# properly configure the server?
# artifacts-destination

# mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri
# sqlite:///mlflow.db --artifacts-destination=artifacts

# Q5. Tune model hyperparameters
# What's the best validation RMSE that you got?
# 5.335

# Q6. Promote the best model to the model registry
# What is the test RMSE of the best model?
# 5.567

mlflow.set_tracking_uri("sqlite:///mlflow.db")
# mlflow.set_tracking_uri(uri="http://localhost:8080")

mlflow.set_experiment("nyc_green_taxis")


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
def run_train(data_path: str):
    """Train a random forest model and log the RMSE metric."""
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():
        mlflow.set_tag("developer", "pedro")
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("max_depth", 10)

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        mlflow.log_param("min_samples_split", rf.min_samples_split)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        with open("models/lin_reg.bin", "wb") as f_out:
            pickle.dump(rf, f_out)

        # mlflow.log_artifact("models/lin_reg.bin", "models")
        mlflow.sklearn.log_model(rf, "models_mlflow")


if __name__ == '__main__':
    run_train()
