from .base_model import BaseModel
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import pickle

class MLPModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = MLPClassifier(**kwargs)

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
