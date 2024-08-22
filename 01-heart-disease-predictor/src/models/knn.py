from .base_model import BaseModel
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class KNNModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = KNeighborsClassifier(**kwargs)
        self.torch_model = None
        self.input_size = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        self.input_size = X.shape[1]
        self._create_torch_model()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_params(self) -> dict:
        return self.model.get_params()

    def _create_torch_model(self):
        class TorchKNN(nn.Module):
            def __init__(self, X_train, y_train, k):
                super(TorchKNN, self).__init__()
                self.X_train = torch.FloatTensor(X_train)
                self.y_train = torch.LongTensor(y_train)
                self.k = k

            def forward(self, x):
                distances = torch.cdist(x, self.X_train)
                _, indices = torch.topk(distances, self.k, largest=False)
                nearest_labels = torch.gather(self.y_train.unsqueeze(0).expand(x.size(0), -1), 1, indices)
                return torch.mode(nearest_labels, dim=1).values

        self.torch_model = TorchKNN(self.model._fit_X, self.model._y, self.model.n_neighbors)

    def to_onnx(self, file_path: str):
        if self.torch_model is None:
            raise ValueError("Model must be fitted before converting to ONNX")

        dummy_input = torch.randn(1, self.input_size)
        torch.onnx.export(self.torch_model, dummy_input, file_path, 
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})
