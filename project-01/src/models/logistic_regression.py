from .base_model import BaseModel
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = Pipeline([
            ('MinMaxScale', MinMaxScaler()),
            ('Logistic', SklearnLogisticRegression(**kwargs))
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_params(self) -> dict:
        return self.model.named_steps['Logistic'].get_params()
