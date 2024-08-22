from typing import Dict
from .base_model import BaseModel
from .catboost import CatBoostModel
from .extra_trees import ExtraTreesModel
from .knn import KNNModel
from .lgbm import LGBMModel
from .logistic_regression import LogisticRegressionModel
from .mlp import MLPModel
from .random_forest import RandomForestModel
from .svm import SVMModel
from .xgboost import XGBoostModel

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

    def save_models(self, models: Dict[str, BaseModel], output_dir: str):
        for name, model in models.items():
            file_path = f"{output_dir}/{name}_model.pkl"
            model.save_model(file_path)
            print(f"Saved {name} model to {file_path}")

    def load_models(self, model_paths: Dict[str, str]) -> Dict[str, BaseModel]:
        models = {}
        for name, path in model_paths.items():
            model_class = globals()[f"{name}Model"]
            model = model_class()
            model.load_model(path)
            models[name] = model
            print(f"Loaded {name} model from {path}")
        return models
