from .base_model import BaseModel
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class MLPModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = MLPClassifier(**kwargs)
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
        if not hasattr(self.model, 'coefs_') or len(self.model.coefs_) == 0:
            print("Warning: MLPClassifier model is not fitted yet. Skipping torch model creation.")
            return

        class TorchMLP(nn.Module):
            def __init__(self, input_size, hidden_layers, output_size):
                super(TorchMLP, self).__init__()
                layers = []
                prev_size = input_size
                for hidden_size in hidden_layers:
                    layers.append(nn.Linear(prev_size, hidden_size))
                    layers.append(nn.ReLU())
                    prev_size = hidden_size
                layers.append(nn.Linear(prev_size, output_size))
                self.layers = nn.Sequential(*layers)

            def forward(self, x):
                return self.layers(x)

        hidden_layer_sizes = self.model.hidden_layer_sizes
        output_size = self.model.n_outputs_
        self.torch_model = TorchMLP(self.input_size, hidden_layer_sizes, output_size)

        # Transfer learned parameters
        for i, layer in enumerate(self.torch_model.layers):
            if isinstance(layer, nn.Linear):
                if i < len(self.model.coefs_):
                    layer.weight.data = torch.FloatTensor(self.model.coefs_[i].T)
                    layer.bias.data = torch.FloatTensor(self.model.intercepts_[i])
                else:
                    print(f"Warning: Mismatch in layer count. Skipping weight transfer for layer {i}")

    def to_onnx(self, file_path: str):
        if self.torch_model is None:
            raise ValueError("Torch model has not been created. Ensure the model is fitted before converting to ONNX")

        dummy_input = torch.randn(1, self.input_size)
        torch.onnx.export(self.torch_model, dummy_input, file_path, 
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})
