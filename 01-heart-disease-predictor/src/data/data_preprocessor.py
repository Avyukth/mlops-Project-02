import pandas as pd
import mlflow
from typing import Tuple
from .feature_preprocessor import FeaturePreprocessor
from .target_preprocessor import TargetPreprocessor

class DataPreprocessor:
    def __init__(self, target_column: str = "heart disease"):
        self.target_column = target_column
        self.feature_preprocessor = FeaturePreprocessor()
        self.target_preprocessor = TargetPreprocessor()
        self.column_mapping = {}

    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        def clean_name(name):
            return name.strip().lower().replace(" ", "_")

        old_columns = df.columns
        new_columns = [clean_name(col) for col in old_columns]
        self.column_mapping = dict(zip(new_columns, old_columns))
        
        df.columns = new_columns
        self.target_column = clean_name(self.target_column)
        
        return df

    def fit(self, df: pd.DataFrame):
        df = self._clean_column_names(df)
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]

        self.feature_preprocessor.fit(X)
        self.target_preprocessor.fit(y)

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = self._clean_column_names(df)
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]

        X_processed = self.feature_preprocessor.transform(X)
        y_processed = self.target_preprocessor.transform(y)

        return X_processed, y_processed

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        self.fit(df)
        return self.transform(df)

    def log_preprocessing_info(self, X: pd.DataFrame, y: pd.Series):
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("num_categorical_features", len(self.feature_preprocessor.cat_cols))
        mlflow.log_param("num_numerical_features", len(self.feature_preprocessor.num_cols))
        mlflow.log_metric("num_samples", len(X))
        mlflow.log_param("target_distribution", dict(y.value_counts(normalize=True)))
        mlflow.log_param("feature_names", list(X.columns))
        mlflow.log_param("column_mapping", self.column_mapping)

    def save(self, feature_filepath: str, target_filepath: str):
        self.feature_preprocessor.save(feature_filepath)
        self.target_preprocessor.save(target_filepath)

    @classmethod
    def load(cls, feature_filepath: str, target_filepath: str):
        preprocessor = cls()
        preprocessor.feature_preprocessor = FeaturePreprocessor.load(feature_filepath)
        preprocessor.target_preprocessor = TargetPreprocessor.load(target_filepath)
        return preprocessor

    def get_original_column_names(self):
        return {v: k for k, v in self.column_mapping.items()}
