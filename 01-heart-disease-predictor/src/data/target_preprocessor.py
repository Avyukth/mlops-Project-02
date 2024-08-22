import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

class TargetPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, y: pd.Series):
        self.label_encoder.fit(y)

    def transform(self, y: pd.Series) -> pd.Series:
        return pd.Series(self.label_encoder.transform(y), index=y.index, name=y.name)

    def fit_transform(self, y: pd.Series) -> pd.Series:
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y: pd.Series) -> pd.Series:
        return pd.Series(self.label_encoder.inverse_transform(y), index=y.index, name=y.name)

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        return joblib.load(filepath)
