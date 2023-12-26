import argparse
from common import read_params
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np


def feature_importance(data, config):
    target = config["base"]["target_column"]
    config = config["feature_selection"]
    drop_columns = config["drop_columns"]
    split = config["split"]
    objective = config["objective"]
    eval_metric = config["eval_metric"]
    eta = config["eta"]
    min_child_weight = config["min_child_weight"]
    subsample = config["subsample"]
    colsample_bytree = config["colsample_bytree"]
    max_depth = config["max_depth"]
    seed = config["seed"]
    nthread = config["nthread"]
    ld = config["lambda"]
    epoches = config["epoches"]
    early_stopping_rounds = config["early_stopping_rounds"]

    X = data.drop(drop_columns, axis=1)
    y = data[target]
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, train_size=split, random_state=0
    )

    dtrain = xgb.DMatrix(xtrain, label=ytrain)
    dvalid = xgb.DMatrix(xtest, label=ytest)
    watchlist = [(dtrain, "train"), (dvalid, "valid")]

    xgb_params = {}
    xgb_params["objective"] = objective
    xgb_params["eval_metric"] = eval_metric
    xgb_params["eta"] = eta
    xgb_params["min_child_weight"] = min_child_weight
    xgb_params["subsample"] = subsample
    xgb_params["colsample_bytree"] = colsample_bytree
    xgb_params["max_depth"] = max_depth
    xgb_params["seed"] = seed
    xgb_params["nthread"] = nthread
    xgb_params["lambda"] = ld

    xgb_model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=epoches,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        maximize=False,
        verbose_eval=0,
    )
    feature_importance = xgb_model.get_score(importance_type="weight")
    threshod = np.mean(list(feature_importance.values()))
    cols = [col for col in feature_importance if feature_importance[col] >= threshod]
    print("Modeling RMSE %.5f" % xgb_model.best_score)
    return data[cols + [target]]


def feature_selection(config_path):
    config = read_params(config_path)
    feature_engineering_data_csv = config["feature_engineering"][
        "feature_engineering_data_csv"
    ]
    feature_selection_data_csv = config["feature_selection"][
        "feature_selection_data_csv"
    ]

    data = pd.read_csv(feature_engineering_data_csv)
    data = feature_importance(data, config)
    data.to_csv(feature_selection_data_csv, index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    feature_selection(config_path=parsed_args.config)
