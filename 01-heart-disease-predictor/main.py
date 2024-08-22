import os
from typing import Dict, List, Tuple

import boto3
import mlflow
import onnx
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from src.config import Config
from src.data.data_manager import DataManager
from src.evaluation.evaluator import Evaluator
from src.models.ensemble_creator import EnsembleCreator
from src.models.model_trainer import ModelTrainer
from src.reporting.evidently_reporter import EvidentlyReporter
from src.utils.mlflow_utils import MLflowUtils
from src.utils.s3_utils import S3Utils

# Initialize configuration
config = Config()
print(config)
# Initialize utilities
s3_utils = S3Utils(config)
mlflow_utils = MLflowUtils(config)

def run_experiment():
    with mlflow_utils.start_run():
        # Data preparation
        data_manager = DataManager(config, s3_utils)
        X, y, cat_cols, num_cols = data_manager.load_and_preprocess_data()
        x_train, x_test, y_train, y_test = data_manager.split_data(X, y)

        # Generate reports
        evidently_reporter = EvidentlyReporter(config)
        report_path, test_suite_path = evidently_reporter.generate_reports(
            x_train, y_train, x_test, y_test
        )
        print(
            "report_path, test_suite_path",
            report_path,
            test_suite_path,
        )
        mlflow_utils.log_artifact(report_path, "evidently_reports")
        mlflow_utils.log_artifact(test_suite_path, "evidently_reports")

        # Log data info
        mlflow_utils.log_data_info(X, cat_cols, num_cols)

        # Train models
        model_trainer = ModelTrainer(config)
        models = model_trainer.train_models(x_train, y_train)

        # Create ensemble models
        ensemble_creator = EnsembleCreator(config)
        ensemble_models = ensemble_creator.create_ensemble_models(
            models, x_train, y_train
        )
        models.update(ensemble_models)

        # Evaluate models
        evaluator = Evaluator(config)
        results, best_model = evaluator.evaluate_models(
            models, x_train, y_train, x_test, y_test
        )

        # Log model results and save models
        for name, result in results.items():
            mlflow_utils.log_metric(f"{name}_train_score", result["train_score"])
            mlflow_utils.log_metric(f"{name}_test_score", result["test_score"])
            
            # Convert and log the model in ONNX format
            initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
            onx = convert_sklearn(models[name], initial_types=initial_type)
            onnx_path = os.path.join(config.PATHS["model_dir"], f"{name}.onnx")
            onnx.save_model(onx, onnx_path)
            mlflow_utils.log_artifact(onnx_path, name)

        # Save and upload best model
        if best_model:
            best_model_path = evaluator.save_best_model(best_model)
            mlflow_utils.log_artifact(best_model_path, "best_model")
            s3_utils.upload_model_artifacts(config.PATHS["model_dir"])

        # Create and upload performance plot
        performance_plot_path = evaluator.create_performance_plot(results)
        mlflow_utils.log_artifact(performance_plot_path, "plots")
        s3_utils.upload_performance_plot(performance_plot_path)

        # Set tags for the run
        mlflow_utils.set_tag("model_version", config.VERSIONS["model"])
        mlflow_utils.set_tag("data_version", config.VERSIONS["data"])


    with mlflow_utils.start_run():
        # Data preparation
        data_manager = DataManager(config, s3_utils)
        X, y, cat_cols, num_cols = data_manager.load_and_preprocess_data()
        x_train, x_test, y_train, y_test = data_manager.split_data(X, y)

        # Generate reports
        evidently_reporter = EvidentlyReporter(config)
        report_path, test_suite_path = evidently_reporter.generate_reports(
            x_train, y_train, x_test, y_test
        )
        print(
            "report_path, test_suite_path",
            report_path,
            test_suite_path,
        )
        mlflow_utils.log_artifact(report_path, "evidently_reports")
        mlflow_utils.log_artifact(test_suite_path, "evidently_reports")

        # Log data info
        mlflow_utils.log_data_info(X, cat_cols, num_cols)

        # Train models
        model_trainer = ModelTrainer(config)
        models = model_trainer.train_models(x_train, y_train)

        # Create ensemble models
        ensemble_creator = EnsembleCreator(config)
        ensemble_models = ensemble_creator.create_ensemble_models(
            models, x_train, y_train
        )
        models.update(ensemble_models)

        # Evaluate models
        evaluator = Evaluator(config)
        results, best_model = evaluator.evaluate_models(
            models, x_train, y_train, x_test, y_test
        )

        # Log model results and save models
        for name, result in results.items():
            mlflow_utils.log_metric(f"{name}_train_score", result["train_score"])
            mlflow_utils.log_metric(f"{name}_test_score", result["test_score"])
            
            # Convert and log the model in ONNX format
            initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
            onx = convert_sklearn(models[name], initial_types=initial_type)
            onnx_path = os.path.join(config.PATHS["model_dir"], f"{name}.onnx")
            onnx.save_model(onx, onnx_path)
            mlflow_utils.log_artifact(onnx_path, name)

        # Save and upload best model
        if best_model:
            best_model_path = evaluator.save_best_model(best_model)
            mlflow_utils.log_artifact(best_model_path, "best_model")
            s3_utils.upload_model_artifacts(config.PATHS["model_dir"])

        # Create and upload performance plot
        performance_plot_path = evaluator.create_performance_plot(results)
        mlflow_utils.log_artifact(performance_plot_path, "plots")
        s3_utils.upload_performance_plot(performance_plot_path)

        # Set tags for the run
        mlflow_utils.set_tag("model_version", config.VERSIONS["model"])
        mlflow_utils.set_tag("data_version", config.VERSIONS["data"])
def main():
    print("Starting Heart Disease Classification project...")

    os.makedirs(config.PATHS["model_dir"], exist_ok=True)
    s3_utils.create_bucket()

    mlflow_utils.setup_mlflow()
    run_experiment()

    print("Heart Disease Classification project completed.")


if __name__ == "__main__":
    main()
