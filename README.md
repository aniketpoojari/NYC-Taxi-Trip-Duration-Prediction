# Project Name

NYC Taxi Trip duration prediction

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact Information](#contact-information)

## Introduction

In our fast-paced world, services like Ola and Uber make getting around easy and affordable. They connect passengers with local drivers through online platforms. The challenge we're facing is improving how taxis are dispatched, specifically predicting how long a driver will be busy with a passenger. If we can accurately estimate when a taxi will be available, it would greatly improve the assignment of drivers to pickup requests.

This project focuses on predicting the total duration of taxi trips in New York City. We're looking for models that can forecast how long a ride will take. The goal is to enhance the efficiency of ride-hailing services and make the experience smoother for both drivers and passengers.

## Features

- [Feature Engineering](src/feature_engineering.py)
- [Feature Selection](src/feature_selection.py)
- [Training](src/training.py)
- [Best Model Selection](src/log_production_model.py)

## Requirements

- yaml
- argparse
- pandas
- numpy
- sklearn
- xgboost
- catboost
- mlflow
- math
- urllib

## Installation

```bash
# Clone the repository
git clone https://github.com/aniketpoojari/NYC-Taxi-Trip-Duration-Prediction.git

# Change directory
cd NYC-Taxi-Trip-Duration-Prediction

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Add your data in the data\raw folder and track it by DVC using the command:
dvc add data\raw\<filename>

# Run mlflow server in the background before running the pipeline
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0

# Change values in the params.yaml file 

# training
dvc repro
```

## Data

- The data is from the [kaggle competetion](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)
- Data fields
    - id - a unique identifier for each trip
    - vendor_id - a code indicating the provider associated with the trip record
    - pickup_datetime - date and time when the meter was engaged
    - dropoff_datetime - date and time when the meter was disengaged
    - passenger_count - the number of passengers in the vehicle (driver entered value)
    - pickup_longitude - the longitude where the meter was engaged
    - pickup_latitude - the latitude where the meter was engaged
    - dropoff_longitude - the longitude where the meter was disengaged
    - dropoff_latitude - the latitude where the meter was disengaged
    - store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
    - trip_duration - duration of the trip in seconds

- Check [notebook](notebooks/NYC-Taxi-Trip-Duration-Prediction.ipynb) to look at all the _Exploratory Data Anlaysis_ and _Experimentations_ done.

## Model Training

```bash
# Train the model
dvc repro
```

## Evaluation

- R2 score is used to evaluate the model

## Results

[Present and interpret the results of the model. Include visualizations if applicable.]
- saved_models folder will contain the final model after the pipeline is executed using MLFlow

## Contributing

[Explain how others can contribute to the project. Include guidelines for reporting issues, suggesting enhancements, and submitting pull requests.]

## License

[Specify the project's license. For example, MIT License.]