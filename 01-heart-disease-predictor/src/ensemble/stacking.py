from typing import List
from sklearn.ensemble import StackingClassifier
from src.models.base_model import BaseModel
import pandas as pd
import numpy as np

class StackingEnsemble(BaseModel):
    def __init__(self, models: List[BaseModel], meta_model: BaseModel = None):
        if meta_model is None:
            from src.models.logistic_regression import LogisticRegressionModel
            meta_model = LogisticRegressionModel()
        
        self.models = models
        self.meta_model = meta_model
        self.ensemble = StackingClassifier(
            estimators=[(f"model_{i}", model.model) for i, model in enumerate(models)],
            final_estimator=meta_model.model
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.ensemble.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.ensemble.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.ensemble.predict_proba(X)

    def get_params(self) -> dict:
        return self.ensemble.get_params()
