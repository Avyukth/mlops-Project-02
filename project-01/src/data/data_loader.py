import pandas as pd
from typing import Dict
import mlflow

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_path)
        self._log_data_info(df)
        return df

    def _log_data_info(self, df: pd.DataFrame) -> None:
        mlflow.log_param("data_source", self.file_path)
        mlflow.log_metric("num_rows", len(df))
        mlflow.log_metric("num_columns", len(df.columns))
