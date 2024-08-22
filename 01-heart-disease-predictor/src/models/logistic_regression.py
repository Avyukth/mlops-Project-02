from .base_model import BaseModel
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)
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
        class TorchLogisticRegression(nn.Module):
            def __init__(self, input_size):
                super(TorchLogisticRegression, self).__init__()
                self.linear = nn.Linear(input_size, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                return self.sigmoid(self.linear(x))

        self.torch_model = TorchLogisticRegression(self.input_size)
        self.torch_model.linear.weight.data = torch.FloatTensor(self.model.coef_)
        self.torch_model.linear.bias.data = torch.FloatTensor(self.model.intercept_)

    def to_onnx(self, file_path: str):
        if self.torch_model is None:
            raise ValueError("Model must be fitted before converting to ONNX")

        dummy_input = torch.randn(1, self.input_size)
        torch.onnx.export(self.torch_model, dummy_input, file_path, 
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})
