from .base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

class RandomForestModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_params(self) -> dict:
        return self.model.get_params()
