from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import pickle

class BaseModel(ABC):
    def __init__(self):
        self.model = None
        self.torch_model = None
        self.input_size = None
        self.label_encoder = LabelEncoder()

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass

    def encode_target(self, y: pd.Series) -> np.ndarray:
        return self.label_encoder.fit_transform(y)

    def decode_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(y_pred)

    def _create_torch_model(self):
        if self.model is None:
            raise ValueError("The underlying model must be fitted before creating a torch model.")
        
        class TorchWrapper(nn.Module):
            def __init__(self, sklearn_model):
                super(TorchWrapper, self).__init__()
                self.sklearn_model = sklearn_model

            def forward(self, x):
                # This is a placeholder implementation and may need to be adjusted
                # based on the specific model type
                return torch.from_numpy(self.sklearn_model.predict_proba(x.numpy()))

        self.torch_model = TorchWrapper(self.model)

    def to_onnx(self, file_path: str) -> None:
        if self.model is None:
            raise ValueError("Model must be fitted before converting to ONNX")

        if self.input_size is None:
            raise ValueError("Input size is not set. Make sure to fit the model first.")

        try:
            initial_type = [('float_input', FloatTensorType([None, self.input_size]))]
            onx = convert_sklearn(self.model, initial_types=initial_type)
            onnx.save_model(onx, file_path)
        except Exception as e:
            print(f"sklearn-onnx conversion failed: {str(e)}. Falling back to PyTorch-based conversion.")
            
            if self.torch_model is None:
                self._create_torch_model()

            dummy_input = torch.randn(1, self.input_size)
            torch.onnx.export(self.torch_model, dummy_input, file_path, 
                              input_names=['input'], output_names=['output'],
                              dynamic_axes={'input': {0: 'batch_size'},
                                            'output': {0: 'batch_size'}})

    @abstractmethod
    def load_model(self, file_path: str) -> None:
        pass

    @abstractmethod
    def save_model(self, file_path: str) -> None:
        pass
