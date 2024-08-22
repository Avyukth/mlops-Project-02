import pytest
import pandas as pd
import numpy as np
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.data.data_splitter import DataSplitter
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.ensemble.voting import VotingEnsemble
from src.evaluation.metrics import evaluate_model
from src.utils.file_utils import load_config
import os
import mlflow
from unittest.mock import patch

@pytest.fixture
def sample_data(tmp_path):
    # Create a sample dataset
    data = pd.DataFrame({
        'age': np.random.randint(30, 80, 100),
        'sex': np.random.choice(['M', 'F'], 100),
        'cp': np.random.randint(0, 4, 100),
        'trestbps': np.random.randint(90, 200, 100),
        'chol': np.random.randint(120, 400, 100),
        'target': np.random.choice([0, 1], 100)
    })
    
    # Save the sample data to a CSV file
    file_path = tmp_path / "sample_heart_disease_data.csv"
    data.to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture
def config_file(tmp_path):
    # Create a sample configuration file
    config = {
        "logistic_regression": {"C": 1.0, "max_iter": 100},
        "random_forest": {"n_estimators": 100, "max_depth": 5}
    }
    file_path = tmp_path / "config.yaml"
    import yaml
    with open(file_path, 'w') as f:
        yaml.dump(config, f)
    return str(file_path)

@patch('mlflow.log_param')
@patch('mlflow.log_metric')
@patch('mlflow.log_artifact')
@patch('mlflow.sklearn.log_model')
def test_full_pipeline(mock_log_model, mock_log_artifact, mock_log_metric, mock_log_param, sample_data, config_file):
    mlflow.set_experiment("test_integration_experiment")
    # Load data
    loader = DataLoader(sample_data)
    df = loader.load_data()
    
    # Preprocess data
    preprocessor = DataPreprocessor(target_column='target')
    X, y, cat_cols, num_cols = preprocessor.preprocess(df)
    
    # Split data
    splitter = DataSplitter(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = splitter.split(X, y)
    
    # Load configuration
    config = load_config(config_file)
    
    # Train models
    lr_model = LogisticRegressionModel(**config['logistic_regression'])
    rf_model = RandomForestModel(**config['random_forest'])
    
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    
    # Create and train ensemble
    ensemble = VotingEnsemble([('lr', lr_model), ('rf', rf_model)])
    ensemble.fit(X_train, y_train)
    
    # Evaluate models
    for name, model in [('Logistic Regression', lr_model), ('Random Forest', rf_model), ('Ensemble', ensemble)]:
        train_score, test_score = evaluate_model(model, X_train, y_train, X_test, y_test)
        print(f"{name} - Train Score: {train_score}, Test Score: {test_score}")
    
    # Assert that mock MLflow logging functions were called
    mock_log_param.assert_called()
    mock_log_metric.assert_called()
    mock_log_artifact.assert_called()
    mock_log_model.assert_called()
    
    # Add more specific assertions as needed
    assert mock_log_model.call_count == 3  # One call for each model
    assert 0 <= train_score <= 1
    assert 0 <= test_score <= 1

if __name__ == "__main__":
    pytest.main([__file__])
