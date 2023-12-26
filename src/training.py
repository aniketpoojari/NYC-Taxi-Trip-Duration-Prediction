import argparse
from common import read_params
from sklearn.model_selection import train_test_split
import pandas as pd
from catboost import CatBoostRegressor
import math
from sklearn.metrics import mean_squared_error, r2_score
import itertools
import mlflow
from urllib.parse import urlparse


def split_data(data, config):
    target = config["base"]["target_column"]
    split = config["training"]["split"]

    X = data.drop(target, axis=1)
    y = data[target]

    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size=split, random_state=0
    )

    return xtrain, xtest, ytrain, ytest


def tuning(data, params):
    cb_model = CatBoostRegressor(**params)

    cb_model.fit(
        data[0],
        data[1],
        eval_set=(data[2], data[3]),
        # cat_features=[4, 6, 7, 8],
        use_best_model=True,
        verbose=False,
    )

    train_val = cb_model.predict(data[0])
    pred_val = cb_model.predict(data[2])

    rmse_score_train = math.sqrt(mean_squared_error(data[1], train_val))
    rmse_score_test = math.sqrt(mean_squared_error(data[3], pred_val))
    r2_score_train = r2_score(data[1], train_val)
    r2_score_test = r2_score(data[3], pred_val)

    msg = "Train RMSE: {:.5f} ".format(rmse_score_train)
    msg += "Valid RMSE: {:.5f}".format(rmse_score_test)
    msg += " Train r2: {:.5f} ".format(r2_score_train)
    msg += "Valid r2: {:.5f}".format(r2_score_test)
    # print(msg)

    results = {
        "rmse_train": rmse_score_train,
        "rmse_test": rmse_score_test,
        "r2_train": r2_score_train,
        "r2_test": r2_score_test,
    }

    return results, cb_model


def training(config_path):
    config = read_params(config_path)
    feature_selection_data_csv = config["feature_selection"][
        "feature_selection_data_csv"
    ]

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    experiment_name = mlflow_config["experiment_name"]
    run_name = mlflow_config["run_name"]
    registered_model_name = mlflow_config["registered_model_name"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)

    training = config["training"]
    iterations = training["iterations"]
    learning_rate = training["learning_rate"]
    depth = training["depth"]
    eval_metric = training["eval_metric"]
    random_seed = training["random_seed"]
    bagging_temperature = training["bagging_temperature"]
    od_type = training["od_type"]
    metric_period = training["metric_period"]
    od_wait = training["od_wait"]
    task_type = training["task_type"]

    data = pd.read_csv(feature_selection_data_csv)
    xtrain, xtest, ytrain, ytest = split_data(data, config)
    data = [xtrain, ytrain, xtest, ytest]

    dic = {
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "eval_metric": eval_metric,
        "random_seed": random_seed,
        "bagging_temperature": bagging_temperature,
        "od_type": od_type,
        "metric_period": metric_period,
        "od_wait": od_wait,
        "task_type": task_type,
    }

    combinations = itertools.product(*list(dic.values()))

    for i in combinations:
        with mlflow.start_run(run_name=run_name) as mlops_run:
            params = {
                "iterations": i[0],
                "learning_rate": i[1],
                "depth": i[2],
                "eval_metric": i[3],
                "random_seed": i[4],
                "bagging_temperature": i[5],
                "od_type": i[6],
                "metric_period": i[7],
                "od_wait": i[8],
                "task_type": i[9],
            }

            results, model = tuning(data, params)

            for p in params:
                mlflow.log_param(p, params[p])

            for r in results:
                mlflow.log_metric(r, results[r])

            tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

            if tracking_url_type_store != "file":
                mlflow.catboost.log_model(
                    model,
                    "model",
                    registered_model_name=registered_model_name,
                )
            else:
                mlflow.catboost.log_model(model, "model")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
