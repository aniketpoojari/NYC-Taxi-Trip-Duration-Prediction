## 📝 Description

At some point or the other almost each one of us has used an Ola or Uber for taking a ride.

Ride hailing services are services that use online-enabled platforms to connect between passengers and local drivers using their personal vehicles. In most cases they are a comfortable method for door-to-door transport. Usually they are cheaper than using licensed taxicabs. Examples of ride hailing services include Uber and Lyft.

To improve the efficiency of taxi dispatching systems for such services, it is important to be able to predict how long a driver will have his taxi occupied. If a dispatcher knew approximately when a taxi driver would be ending their current ride, they would be better able to identify which driver to assign to each pickup request.

In this competition, we are challenged to build a model that predicts the total ride duration of taxi trips in New York City.

## :test_tube: Experimentation
- Check notebooks folder to look at all the Exploratory Data Anlaysis Done
- Check notebooks folder to see all the experientation done before creating the final pipeline

## Pipeline
- DVC is used to create pipeline
- Change the values in the params.yaml to test different values that effest the model

## :gear: Requirements
- Use ```pip install -r requirements.txt``` to install the requirements

## :runner: How to run
- Use ```dvc add data\raw\<filename>``` to make dvc track the input file
- Use ```mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0``` in the project directory to run mlflow server in the background before running the pipeline
- Use ```dvc repro``` to run the pipeline

## :robot: Final Model
- saved_models folder will contain the final model after the pipeline is executed using MLFlow