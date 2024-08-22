import pytest
from src.evaluation.metrics import evaluate_model
from src.models.logistic_regression import LogisticRegressionModel
import pandas as pd
import numpy as np
from unittest.mock import patch
import mlflow

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5] * 20,
        'feature2': [5, 4, 3, 2, 1] * 20
    })
    y = pd.Series([0, 0, 1, 1, 1] * 20)
    return X, y

@patch('mlflow.log_metric')
@patch('mlflow.log_artifact')
def test_evaluate_model(mock_log_artifact, mock_log_metric, sample_data):
    X, y = sample_data
    model = LogisticRegressionModel()
    
    # Set up an MLflow experiment
    experiment_name = "test_experiment"
    mlflow.set_experiment(experiment_name)
    
    # Ensure we start an MLflow run
    with mlflow.start_run():
        model.fit(X, y)
        train_score, test_score = evaluate_model(model, X, y, X, y)
    
    assert isinstance(train_score, float)
    assert isinstance(test_score, float)
    assert 0 <= train_score <= 1
    assert 0 <= test_score <= 1

    mock_log_metric.assert_called()
    mock_log_artifact.assert_called()
