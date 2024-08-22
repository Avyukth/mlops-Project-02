from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from src.models.base_model import BaseModel
import pandas as pd
import numpy as np

class SVMModel(BaseModel):
    def __init__(self, **kwargs):
        # Remove 'probability' from kwargs if it's there, as we're setting it explicitly
        kwargs.pop('probability', None)
        self.model = Pipeline([
            ('MinMaxScale', MinMaxScaler()),
            ('SVC', SVC(probability=True, **kwargs))
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_params(self) -> dict:
        return self.model.named_steps['SVC'].get_params()
