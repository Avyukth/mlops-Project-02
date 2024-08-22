import pytest
from src.data.data_preprocessor import DataPreprocessor
import pandas as pd
import numpy as np
from unittest.mock import patch

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'numeric1': [1, 2, 3, 4],
        'numeric2': [5.5, 6.5, 7.5, 8.5],
        'categorical': ['A', 'B', 'A', 'C'],
        'target': [0, 1, 1, 0]
    })

@patch('mlflow.log_param')
@patch('mlflow.log_metric')
@patch('mlflow.log_artifact')
def test_data_preprocessor(mock_log_artifact, mock_log_metric, mock_log_param, sample_data):
    preprocessor = DataPreprocessor(target_column='target')
    X, y, cat_cols, num_cols = preprocessor.preprocess(sample_data)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(cat_cols, list)
    assert isinstance(num_cols, list)

    # Check that we have the correct number of rows
    assert X.shape[0] == 4

    # Check that we have the correct columns
    expected_columns = ['numeric1', 'numeric2', 'categorical_A', 'categorical_B', 'categorical_C']
    assert all(col in X.columns for col in expected_columns)

    # Check that y has the correct shape
    assert y.shape == (4,)

    # Check that categorical and numerical columns are correctly identified
    assert set(cat_cols) == {'categorical'}
    assert set(num_cols) == {'numeric1', 'numeric2'}

    mock_log_param.assert_called()
    mock_log_metric.assert_called()
    mock_log_artifact.assert_not_called()  # We're not logging any artifacts in this method
