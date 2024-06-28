from datetime import datetime

import pandas as pd

from batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID',
               'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    df = prepare_data(df, ['PULocationID', 'DOLocationID'])

    expected_data = [
        (-1, -1, dt(1, 2), dt(1, 10), 8),
        (1, -1, dt(1, 2), dt(1, 10), 8),
        (1, 2, dt(2, 2), dt(2, 3), 1)
    ]
    expected_df = pd.DataFrame(expected_data, columns=columns + ['duration'])
    expected_df['PULocationID'] = expected_df['PULocationID'].astype(str)
    expected_df['DOLocationID'] = expected_df['DOLocationID'].astype(str)
    expected_df['duration'] = expected_df['duration'].astype(float)
    assert df.equals(expected_df), "The DataFrame is not as expected."
