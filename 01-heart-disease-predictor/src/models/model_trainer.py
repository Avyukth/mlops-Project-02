from typing import Dict
from src.models.catboost import CatBoostModel
from src.models.extra_trees import ExtraTreesModel
from src.models.knn import KNNModel
from src.models.lgbm import LGBMModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.mlp import MLPModel
from src.models.random_forest import RandomForestModel
from src.models.svm import SVMModel
from src.models.xgboost import XGBoostModel
from src.models.base_model import BaseModel

class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train_models(self, x_train, y_train) -> Dict[str, BaseModel]:
        models = {
            "Logistic": LogisticRegressionModel(**self.config.MODEL_PARAMS['logistic_regression']),
            "SVM": SVMModel(**self.config.MODEL_PARAMS['svm']),
            "KNN": KNNModel(**self.config.MODEL_PARAMS['knn']),
            "MLP": MLPModel(**self.config.MODEL_PARAMS['mlp']),
            "RandomForest": RandomForestModel(**self.config.MODEL_PARAMS['random_forest']),
            "ExtraTrees": ExtraTreesModel(**self.config.MODEL_PARAMS['extra_trees']),
            "CatBoost": CatBoostModel(**self.config.MODEL_PARAMS['catboost']),
            "LGBM": LGBMModel(**self.config.MODEL_PARAMS['lgbm']),
            "XGB": XGBoostModel(**self.config.MODEL_PARAMS['xgboost'])
        }

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(x_train, y_train)

        return models

    def export_models_to_onnx(self, models: Dict[str, BaseModel], output_dir: str):
        for name, model in models.items():
            onnx_file_path = f"{output_dir}/{name.lower()}_model.onnx"
            model.to_onnx(onnx_file_path)
            print(f"Exported {name} model to ONNX format: {onnx_file_path}")
