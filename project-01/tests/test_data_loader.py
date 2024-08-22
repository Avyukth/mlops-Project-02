import pytest
from src.data.data_loader import DataLoader
import pandas as pd
import mlflow
from unittest.mock import patch

@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [0, 1, 0]
    })
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    return file_path

@patch('mlflow.log_param')
@patch('mlflow.log_metric')
def test_data_loader(mock_log_metric, mock_log_param, sample_csv):
    loader = DataLoader(sample_csv)
    df = loader.load_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert list(df.columns) == ['feature1', 'feature2', 'target']
    mock_log_param.assert_called()
    mock_log_metric.assert_called()
