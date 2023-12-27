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

[List key features and functionalities of the project.]

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

[Provide information about the data used in the project. Include the source, format, and any preprocessing steps.]
- Check notebooks folder to look at all the Exploratory Data Anlaysis Done
- Check notebooks folder to see all the experientation done before creating the final pipeline

## Model Training

[Explain how to train the machine learning model. Include details about the algorithm, hyperparameters, and any other relevant information.]

```bash
# Train the model
python train.py --data train_data.csv --model saved_model.pkl
```

## Evaluation

[Describe the metrics and methods used to evaluate the model.]

## Results

[Present and interpret the results of the model. Include visualizations if applicable.]
- saved_models folder will contain the final model after the pipeline is executed using MLFlow

## Contributing

[Explain how others can contribute to the project. Include guidelines for reporting issues, suggesting enhancements, and submitting pull requests.]

## License

[Specify the project's license. For example, MIT License.]

## Contact Information

[Provide contact information for the project maintainer or team.]

---

Feel free to customize this template based on the specific details of your project. Remember to keep the README concise, well-organized, and easy to follow.