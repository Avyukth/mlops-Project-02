
# Heart Disease Predictor App

## Overview

This application is a machine learning-based heart disease predictor. It uses various classification models to predict the likelihood of heart disease based on patient data. The app includes data processing, model training, and evaluation components, all tracked using MLflow for experiment management.

## Features

- Data preprocessing and feature engineering
- Multiple classification models including:
  - Logistic Regression
  - Support Vector Classifier (SVC)
  - K-Nearest Neighbors (KNN)
  - Multi-Layer Perceptron (MLP)
  - Random Forest
  - Extra Trees
  - CatBoost
  - LightGBM (LGBM)
  - XGBoost (XGB)
- Ensemble methods:
  - Voting Classifier (Soft and Hard voting)
  - Stacking Classifier
- Hyperparameter tuning using GridSearchCV
- Model evaluation and comparison
- MLflow tracking for experiment management
- Dockerized application for easy deployment

## Project Structure

## Installation

1. Clone the repository:
---
```
   git clone https://github.com/yourusername/heart-disease-predictor.git
   cd heart-disease-predictor
```
---

2. Build and run the Docker containers:
---
```
   make all
```
---

## Usage

The application is managed using Docker Compose and a Makefile. Here are the available commands:

- `make up`: Start Docker Compose services in detached mode
- `make down`: Stop and remove Docker Compose services
- `make rmi`: Remove all Docker images (if any exist)
- `make all`: Run all commands in reverse order (down, rmi, up)

To run the heart disease prediction:

1. Ensure your data is in the `data/` directory as `dataset_heart.csv`.
2. Run the main script:
---
```
   docker-compose run app python main.py
```
---

<!-- ## Data Processing

The `data_process.py` script handles the following tasks:

1. Reading the dataset
2. Summarizing data statistics
3. Splitting features into categorical and numerical
4. Preprocessing the data (encoding categorical variables, etc.)
5. Preparing train-test splits

All these steps are tracked using MLflow for reproducibility.

## Model Training and Evaluation

The `main.py` script performs the following:

1. Sets up MLflow tracking
2. Creates model pipelines for various classifiers
3. Performs grid search for hyperparameter tuning
4. Creates ensemble models (Voting and Stacking Classifiers)
5. Evaluates models and saves results
6. Generates performance comparison plots

## Results

Model results are saved in `models/model_results.json`. Performance plots are generated and saved in the `models/` directory.

## MLflow Tracking

MLflow is used to track experiments, including:

- Data source and statistics
- Model parameters
- Evaluation metrics
- Artifacts (data summaries, preprocessed data, model files)

To view the MLflow UI, run:
---
```
mlflow ui
```
---
Then open a browser and navigate to `http://localhost:5000`. -->

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- MLflow: [https://mlflow.org/](https://mlflow.org/)
