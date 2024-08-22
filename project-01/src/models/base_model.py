from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

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
