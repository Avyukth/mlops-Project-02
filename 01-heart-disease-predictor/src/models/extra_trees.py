from .base_model import BaseModel
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class ExtraTreesModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = ExtraTreesClassifier(**kwargs)
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
        class TorchExtraTrees(nn.Module):
            def __init__(self, input_size, n_trees):
                super(TorchExtraTrees, self).__init__()
                self.trees = nn.ModuleList([nn.Linear(input_size, 1) for _ in range(n_trees)])

            def forward(self, x):
                return torch.mean(torch.stack([tree(x) for tree in self.trees]), dim=0)

        n_trees = len(self.model.estimators_)
        self.torch_model = TorchExtraTrees(self.input_size, n_trees)
        # Transfer learned parameters (this is a placeholder and needs to be implemented properly)

    def to_onnx(self, file_path: str):
        if self.torch_model is None:
            raise ValueError("Model must be fitted before converting to ONNX")

        dummy_input = torch.randn(1, self.input_size)
        torch.onnx.export(self.torch_model, dummy_input, file_path, 
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})
