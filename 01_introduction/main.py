"""Code for Homework 1 of MLOps Zoomcamp 2024."""
import gc

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Read the data for January
df_january = pd.read_parquet("data/yellow_tripdata_2023-01.parquet")
# Read the data for January. How many columns are there?
n_cols = len(df_january.columns)
# 19
print(f"Number of columns in January data: {n_cols}")

# What's the standard deviation of the trips duration in January?
df_january["duration"] = (
    df_january["tpep_dropoff_datetime"] -
    df_january["tpep_pickup_datetime"]).dt.total_seconds()
df_january["duration"] = df_january["duration"] / 60

std_duration = df_january["duration"].std()
# 42.59
print(f"Standard deviation of trip duration in January: {std_duration}")

# Next, we need to check the distribution of the duration variable.
# There are some outliers. Let's remove them and keep only the records where
# the duration was between 1 and 60 minutes (inclusive).
# What fraction of the records left after you dropped the outliers?
original_n_rows = df_january.shape[0]

df_january = df_january[(df_january["duration"] >= 1) &
                        (df_january["duration"] <= 60)]
new_n_rows = df_january.shape[0]

fraction_left = new_n_rows / original_n_rows
# 0.981
print(f"Fraction of records left after dropping outliers: {fraction_left}")

# Let's apply one-hot encoding to the pickup and dropoff location IDs.
# We'll use only these two features for our model.
# Turn the dataframe into a list of dictionaries (remember to re-cast the ids
# to strings - otherwise it will label encode them)
# Fit a dictionary vectorizer
# Get a feature matrix from it
# What's the dimensionality of this matrix (number of columns)?
# 515
df_january["PULocationID"] = df_january["PULocationID"].astype(str)
df_january["DOLocationID"] = df_january["DOLocationID"].astype(str)

df_train = df_january[["PULocationID", "DOLocationID", "duration"]]
df_train["duration"] = df_train["duration"].astype(np.int16)
del df_january
gc.collect()
enc = OneHotEncoder(handle_unknown='ignore', dtype=np.int16)
encoded_data = enc.fit_transform(
    df_train[["PULocationID", "DOLocationID"]]).toarray()
encoded_data = encoded_data.astype(np.int8)
y_train = df_train["duration"].values
del df_train
gc.collect()

# Now let's use the feature matrix from the previous step to train a model.
# Train a plain linear regression model with default parameters
# Calculate the RMSE of the model on the training data
# What's the RMSE on train?
reg = LinearRegression().fit(encoded_data, y_train)
predictions = reg.predict(encoded_data)
mse = mean_squared_error(y_train, predictions)
rmse = mse**0.5
print(f"RMSE on train: {rmse}")

# Now let's apply this model to the validation dataset (February 2023).
# What's the RMSE on validation?

df_february = pd.read_parquet("data/yellow_tripdata_2023-02.parquet")
df_february["duration"] = (
    df_february["tpep_dropoff_datetime"] -
    df_february["tpep_pickup_datetime"]).dt.total_seconds()
df_february["duration"] = df_february["duration"] / 60

df_february["PULocationID"] = df_february["PULocationID"].astype(str)
df_february["DOLocationID"] = df_february["DOLocationID"].astype(str)
df_val = df_february[["PULocationID", "DOLocationID", "duration"]]
df_val["duration"] = df_val["duration"].astype(np.int16)
del df_february
gc.collect()

y_val = df_val["duration"].values
encoded_data_val = enc.transform(
    df_val[["PULocationID", "DOLocationID"]]).toarray()
predictions_val = reg.predict(encoded_data_val)
del df_val
gc.collect()
mse_val = mean_squared_error(y_val, predictions_val)
rmse_val = mse_val**0.5
print(f"RMSE on validation: {rmse_val}")
