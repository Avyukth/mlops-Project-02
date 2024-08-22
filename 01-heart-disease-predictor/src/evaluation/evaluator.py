import os
from typing import Dict, Tuple

import cloudpickle
import joblib
import numpy as np
import onnx
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.evaluation.visualizations import create_performance_plot
from src.models.base_model import BaseModel


class Evaluator:
    def __init__(self, config):
        self.config = config

    def evaluate_models(
        self,
        models: Dict[str, BaseModel],
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[Dict[str, Dict[str, float]], BaseModel]:
        results = {}
        best_score = 0
        best_model = None

        for name, model in models.items():
            train_score = self.evaluate_model(model, x_train, y_train)
            test_score = self.evaluate_model(model, x_test, y_test)
            results[name] = {"train_score": train_score, "test_score": test_score}

            print(
                f"{name} - Train Score: {train_score:.4f}, Test Score: {test_score:.4f}"
            )

            if test_score > best_score:
                best_score = test_score
                best_model = model

        return results, best_model

    def evaluate_model(self, model: BaseModel, X: pd.DataFrame, y: pd.Series) -> float:
        y_pred = model.predict(X)
        return accuracy_score(y, y_pred)

    def save_best_model(self, model: BaseModel) -> str:
        best_model_path = os.path.join(
            self.config.PATHS["model_dir"], "best_model.onnx"
        )

        # Convert the model to ONNX
        initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
        onx = convert_sklearn(model, initial_types=initial_type)

        # Save the ONNX model
        onnx.save_model(onx, best_model_path)

        return best_model_path

    def create_performance_plot(self, results: Dict[str, Dict[str, float]]) -> str:
        plot_path = os.path.join(self.config.PATHS["model_dir"], "model_comparison.png")
        create_performance_plot(results, plot_path)
        return plot_path
