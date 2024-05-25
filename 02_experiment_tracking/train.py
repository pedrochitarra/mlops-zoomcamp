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

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc_green_taxis")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

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


if __name__ == '__main__':
    run_train()
