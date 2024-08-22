# src/data/data_splitter.py

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
import mlflow

class DataSplitter:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the data into training and testing sets.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

        self._log_split_info(X_train, X_test)

        return X_train, X_test, y_train, y_test

    def _log_split_info(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
        """
        Log information about the data split to MLflow.

        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Testing features
        """
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("test_ratio", self.test_size)
        mlflow.log_param("random_state", self.random_state)
