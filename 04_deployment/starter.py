import sys

import pickle
import pandas as pd


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


year = int(sys.argv[1])
month = int(sys.argv[2])

filename = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_"
    f"tripdata_{year:04d}-{month:02d}.parquet")

print(f"Reading data from: {filename}")

df = read_data(filename)

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

print(f"The standard deviation is: {y_pred.std()}")
print(f"The mean prediction is: {y_pred.mean()}")

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
df_result = df[['ride_id']].copy()
df_result['prediction'] = y_pred

output_file = f'predictions_{year:04d}_{month:02d}.parquet'
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)
