from .base_model import BaseModel
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np

class LGBMModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = LGBMClassifier(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_params(self) -> dict:
        return self.model.get_params()
