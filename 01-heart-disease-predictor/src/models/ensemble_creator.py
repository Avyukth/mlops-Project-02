from typing import Dict, List
import pandas as pd
from .base_model import BaseModel
from sklearn.ensemble import VotingClassifier, StackingClassifier
import pickle
import numpy as np

class EnsembleCreator:
    def __init__(self, config):
        self.config = config

    def create_ensemble_models(self, models: Dict[str, BaseModel], x_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, BaseModel]:
        estimators = [(name, model.model) for name, model in models.items()]
        
        voting_soft = VotingClassifier(estimators=estimators, voting='soft')
        voting_hard = VotingClassifier(estimators=estimators, voting='hard')
        stacking_logistic = StackingClassifier(estimators=estimators, final_estimator=models['Logistic'].model)
        stacking_lgbm = StackingClassifier(estimators=estimators, final_estimator=models['LGBM'].model)

        ensemble_models = {
            "Ensemble_Soft": voting_soft,
            "Ensemble_Hard": voting_hard,
            "Stacking_Logistic": stacking_logistic,
            "Stacking_LGBM": stacking_lgbm
        }

        # Fit the ensemble models
        for name, model in ensemble_models.items():
            print(f"Fitting {name}...")
            model.fit(x_train, y_train)

        # Wrap ensemble models in BaseModel
        wrapped_models = {}
        for name, model in ensemble_models.items():
            wrapped_model = BaseModelWrapper(model)
            wrapped_model.fit(x_train, y_train)
            wrapped_models[name] = wrapped_model

        return wrapped_models

class BaseModelWrapper(BaseModel):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        y_encoded = self.encode_target(y)
        self.model.fit(X, y_encoded)
        self.input_size = X.shape[1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_pred_encoded = self.model.predict(X)
        return self.decode_predictions(y_pred_encoded)

    def get_params(self) -> dict:
        return self.model.get_params()

    def save_model(self, file_path: str) -> None:
        with open(file_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'input_size': self.input_size
            }, f)

    def load_model(self, file_path: str) -> None:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.input_size = data['input_size']
