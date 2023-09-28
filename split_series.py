from more_utils.service import TimeseriesGenerator

generator = TimeseriesGenerator()


import pandas as pd

df = pd.read_csv('./data/eugene.csv')


print(df.columns)

import os
import pathlib
df = generator.split_time_series_by_features(
    input_file_path="./data/eugene.csv",
    timestamp_column=df.columns[0],
    features=df.columns[1:],
    output_location='./temp_data',
    delimiter=","
)
