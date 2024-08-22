from .base_model import BaseModel
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class SVMModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = SVC(**kwargs)
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
        class TorchSVM(nn.Module):
            def __init__(self, input_size, n_support_vectors):
                super(TorchSVM, self).__init__()
                self.weight = nn.Parameter(torch.randn(n_support_vectors, input_size))
                self.bias = nn.Parameter(torch.randn(1))

            def forward(self, x):
                kernel = torch.mm(x, self.weight.t())
                return torch.sign(torch.sum(kernel, dim=1) + self.bias)

        n_support_vectors = self.model.n_support_.sum()
        self.torch_model = TorchSVM(self.input_size, n_support_vectors)
        
        # Transfer learned parameters
        support_vectors = self.model.support_vectors_
        dual_coef = self.model.dual_coef_
        self.torch_model.weight.data = torch.FloatTensor(support_vectors * dual_coef.T)
        self.torch_model.bias.data = torch.FloatTensor([self.model.intercept_[0]])

    def to_onnx(self, file_path: str):
        if self.torch_model is None:
            raise ValueError("Model must be fitted before converting to ONNX")

        dummy_input = torch.randn(1, self.input_size)
        torch.onnx.export(self.torch_model, dummy_input, file_path, 
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})
