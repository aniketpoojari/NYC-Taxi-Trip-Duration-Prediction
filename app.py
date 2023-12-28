from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
# run_with_ngrok(app)

# LOADING PRETRAINED MODEL
cb_model = CatBoostRegressor()
cb_model.load_model("saved_models\model.cb")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    features = [[]]
    features[0].append(float(request.form["Pickup Longitude"]))
    features[0].append(float(request.form["Pickup Latitude"]))
    features[0].append(float(request.form["Dropoff Longitude"]))
    features[0].append(float(request.form["Dropoff Latitude"]))

    date = " ".join(request.form["Date Time"].split("T")) + ":00"
    date = pd.to_datetime(date)

    features[0].append(date.hour)

    dlon = float(request.form["Pickup Longitude"]) - float(
        request.form["Dropoff Longitude"]
    )
    dlat = float(request.form["Pickup Latitude"]) - float(
        request.form["Dropoff Latitude"]
    )
    a = (np.sin(dlat / 2)) ** 2 + np.cos(
        float(request.form["Pickup Latitude"])
    ) * np.cos(float(request.form["Dropoff Latitude"])) * (np.sin(dlon / 2)) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    features[0].append(6373.0 * c)

    features[0].append(date.weekofyear)
    features[0].append(date.minute)
    features[0].append(date.weekday() * 24 + date.hour)

    # PREDICTION FROM SAVED MODEL
    result = cb_model.predict(features)

    # SHOWING RESULT TO USER
    return render_template(
        "index.html",
        prediction_text="Trip Duration = {:.2f} Min".format(result[0] / 60),
    )


if __name__ == "__main__":
    app.run()
