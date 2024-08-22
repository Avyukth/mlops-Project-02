from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import torch

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass

    @abstractmethod
    def to_onnx(self, file_path: str) -> None:
        pass

    def _create_torch_model(self):
        pass
