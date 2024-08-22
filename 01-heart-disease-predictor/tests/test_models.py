import pytest
from src.models.logistic_regression import LogisticRegressionModel
from src.models.svm import SVMModel
from src.models.knn import KNNModel
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1]
    })
    y = pd.Series([0, 0, 1, 1, 1])
    return X, y

def test_logistic_regression(sample_data):
    X, y = sample_data
    model = LogisticRegressionModel()
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert all(isinstance(pred, (np.int64, int)) for pred in predictions)

def test_svm(sample_data):
    X, y = sample_data
    model = SVMModel()
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert all(isinstance(pred, (np.int64, int)) for pred in predictions)

def test_knn(sample_data):
    X, y = sample_data
    model = KNNModel()
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert all(isinstance(pred, (np.int64, int)) for pred in predictions)
