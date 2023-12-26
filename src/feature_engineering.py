import argparse
from common import read_params
import pandas as pd
import numpy as np


def dropping_columns(data, drop_columns):
    data = data.drop(drop_columns, axis=1)
    return data


def date_based_features(data):
    data["pickup_datetime"] = pd.to_datetime(data.pickup_datetime)
    data["month"] = data["pickup_datetime"].dt.month
    data["day_of_week"] = data["pickup_datetime"].dt.weekday
    data["hour_of_day"] = data["pickup_datetime"].dt.hour
    data["pickup_hour_weekofyear"] = data["pickup_datetime"].dt.weekofyear
    data["pickup_minute"] = data["pickup_datetime"].dt.minute
    data["pickup_week_hour"] = data["day_of_week"] * 24 + data["hour_of_day"]
    data["day_of_week_based_level"] = data["day_of_week"].apply(
        lambda x: 0 if x in [0, 6] else 1
    )
    return data


def removing_outliers(data, cont_cols):
    for i in cont_cols:
        IQR = data[i].quantile(0.75) - data[i].quantile(0.25)
        MAX = data[i].quantile(0.75) + (1.5 * IQR)
        MIN = data[i].quantile(0.25) - (1.5 * IQR)
        data = data[(data[i] > MIN) & (data[i] < MAX)]
    return data


def passenger_count_based_features(data):
    data["passenger_count_0/1"] = data["passenger_count"].apply(
        lambda x: 0 if x == 0 else 1
    )
    data["passenger_count_by_count"] = data["passenger_count"].map(
        {0: 0, 1: 2, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0}
    )
    return data


def distance_based_features(data):
    dlon = data["pickup_longitude"] - data["dropoff_longitude"]
    dlat = data["pickup_latitude"] - data["dropoff_latitude"]
    a = (np.sin(dlat / 2)) ** 2 + np.cos(data["pickup_latitude"]) * np.cos(
        data["dropoff_latitude"]
    ) * (np.sin(dlon / 2)) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    data["distance"] = 6373.0 * c
    data["l50distance"] = np.log(data["distance"] + 50)
    return data


def vendor_based_features(data):
    data["vendor_id"] = data["vendor_id"] - 1
    return data


def binning(data):
    data["pickup_latitude_round3"] = np.round(data["pickup_latitude"], 3)
    data["pickup_longitude_round3"] = np.round(data["pickup_longitude"], 3)
    data["dropoff_latitude_round3"] = np.round(data["dropoff_latitude"], 3)
    data["dropoff_longitude_round3"] = np.round(data["dropoff_longitude"], 3)
    return data


def feature_engineering(config_path):
    config = read_params(config_path)
    raw_data_csv = config["get_data"]["raw_data_csv"]
    feature_engineering_data_csv = config["feature_engineering"][
        "feature_engineering_data_csv"
    ]
    drop_columns = config["feature_engineering"]["drop_columns"]
    cont_cols = config["feature_engineering"]["cont_cols"]

    data = pd.read_csv(raw_data_csv)
    data = dropping_columns(data, drop_columns)
    data = date_based_features(data)
    data = removing_outliers(data, cont_cols)
    data = passenger_count_based_features(data)
    data = distance_based_features(data)
    data = vendor_based_features(data)
    data = binning(data)
    data.to_csv(feature_engineering_data_csv, index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    feature_engineering(config_path=parsed_args.config)
