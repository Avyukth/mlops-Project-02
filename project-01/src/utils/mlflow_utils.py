import os
from typing import Any, Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient
from src.config import Config


class MLflowUtils:
    def __init__(self, config: Config):
        """
        Initialize MLflowUtils with configuration.

        Args:
            config (Config): Configuration object containing MLflow settings.
        """
        self.config = config
        self.tracking_uri = config["MLFLOW_TRACKING_URI"]
        self.experiment_name = config.APP["experiment_name"]
        self.client = None
        self.experiment = None
        self.default_artifact_root = config.get(
            "MLFLOW_DEFAULT_ARTIFACT_ROOT", "file:///app/mlruns"
        )

    def setup_mlflow(self) -> str:
        """
        Set up MLflow tracking.

        Returns:
            str: Experiment ID
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()

        # Check if experiment exists, create if it doesn't
        self.experiment = self.client.get_experiment_by_name(self.experiment_name)
        if not self.experiment:
            experiment_id = self.client.create_experiment(
                name=self.experiment_name, artifact_location=self.default_artifact_root
            )
            self.experiment = self.client.get_experiment(experiment_id)

        mlflow.set_experiment(self.experiment_name)

        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"MLflow artifact URI: {mlflow.get_artifact_uri()}")

        return self.experiment.experiment_id

    def log_model_params(self, model: Any):
        """
        Log model parameters to MLflow.

        Args:
            model (Any): The trained model
        """
        params = model.get_params()
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

    def log_model(self, model: Any, model_name: str):
        """
        Log the model to MLflow.

        Args:
            model (Any): The trained model
            model_name (str): Name of the model
        """
        mlflow.sklearn.log_model(model, model_name)

    def log_metric(self, key: str, value: float):
        """
        Log a metric to MLflow.

        Args:
            key (str): Metric name
            value (float): Metric value
        """
        mlflow.log_metric(key, value)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact to MLflow.

        Args:
            local_path (str): Path to the local file to be logged as an artifact
            artifact_path (str, optional): Path within the artifact directory to log the file
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_data_info(self, X: Any, cat_cols: list, num_cols: list):
        """
        Log data information to MLflow.

        Args:
            X (Any): Feature DataFrame
            cat_cols (list): List of categorical column names
            num_cols (list): List of numerical column names
        """
        mlflow.log_param("num_samples", len(X))
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("num_categorical_features", len(cat_cols))
        mlflow.log_param("num_numerical_features", len(num_cols))

    def start_run(self):
        """
        Start a new MLflow run.

        Returns:
            mlflow.ActiveRun: The active MLflow run
        """
        active_run = mlflow.active_run()
        if active_run is not None:
            print(f"Ending active run: {active_run.info.run_id}")
            mlflow.end_run()
        return mlflow.start_run()

    def end_run(self):
        """
        End the current MLflow run.
        """
        mlflow.end_run()

    def get_tracking_uri(self) -> str:
        """
        Get the current tracking URI.

        Returns:
            str: The current tracking URI
        """
        return mlflow.get_tracking_uri()

    def set_tag(self, key: str, value: str):
        """
        Set a tag for the current run.

        Args:
            key (str): Tag name
            value (str): Tag value
        """
        mlflow.set_tag(key, value)
