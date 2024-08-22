from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.preprocessor = None
        self.cat_cols = None
        self.num_cols = None

    def fit(self, X, y=None):
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', self.num_cols),
                ('cat', categorical_transformer, self.cat_cols)
            ])

        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)

    def get_feature_names_out(self):
        return self.preprocessor.get_feature_names_out()

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        return joblib.load(filepath)
