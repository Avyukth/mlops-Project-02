import json
import os
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from data_process import main as process_data
from lightgbm import LGBMClassifier
from mlflow.tracking import MlflowClient
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Constants
MODEL_DIR = "./models"
RESULTS_FILE = f"{MODEL_DIR}/model_results.json"
DATA_SPLITS_FILE = f"{MODEL_DIR}/data_splits.joblib"


def setup_mlflow(experiment_name: str) -> Tuple[MlflowClient, str]:
    """Set up MLflow tracking."""
    mlflow.set_tracking_uri("sqlite:///data/mlflow.db")
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    return client, experiment.experiment_id


def create_pipelines() -> Dict[str, Pipeline]:
    """Create model pipelines."""
    return {
        "Logistic": Pipeline(
            [
                ("MinMaxScale", MinMaxScaler()),
                ("Logistic", LogisticRegression(solver="liblinear", max_iter=1000)),
            ]
        ),
        "SVC": Pipeline(
            [("MinMaxScale", MinMaxScaler()), ("SVC", SVC(probability=True))]
        ),
        "KNN": Pipeline(
            [("MinMaxScale", MinMaxScaler()), ("KNN", KNeighborsClassifier())]
        ),
        "MLP": Pipeline(
            [("MinMaxScale", MinMaxScaler()), ("MLP", MLPClassifier(max_iter=1000))]
        ),
        "RandomForest": Pipeline([("RandomForest", RandomForestClassifier())]),
        "ExtraTrees": Pipeline([("ExtraTrees", ExtraTreesClassifier())]),
        "CatBoost": Pipeline([("CatBoost", CatBoostClassifier(verbose=0))]),
        "LGBM": Pipeline([("LGBM", LGBMClassifier(verbosity=-1))]),
        "XGB": Pipeline([("XGB", XGBClassifier())]),
    }


def create_param_grids() -> Dict[str, Dict]:
    """Create parameter grids for grid search."""
    return {
        "Logistic": {"Logistic__C": [0.1, 1, 10], "Logistic__penalty": ["l1", "l2"]},
        "SVC": {"SVC__C": [0.1, 1, 10], "SVC__kernel": ["linear", "rbf"]},
        "KNN": {"KNN__n_neighbors": [3, 5, 7], "KNN__weights": ["uniform", "distance"]},
        "MLP": {
            "MLP__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "MLP__activation": ["relu", "tanh"],
            "MLP__alpha": [0.001, 0.01, 0.1],
        },
        "RandomForest": {
            "RandomForest__n_estimators": [100, 200, 300],
            "RandomForest__max_depth": [None, 5, 10],
        },
        "ExtraTrees": {
            "ExtraTrees__n_estimators": [100, 200, 300],
            "ExtraTrees__max_depth": [None, 5, 10],
        },
        "CatBoost": {
            "CatBoost__iterations": [100, 200, 300],
            "CatBoost__depth": [6, 8, 10],
        },
        "LGBM": {
            "LGBM__n_estimators": [100, 200, 300],
            "LGBM__max_depth": [None, 5, 10],
        },
        "XGB": {"XGB__n_estimators": [100, 200, 300], "XGB__max_depth": [None, 5, 10]},
    }


def perform_grid_search(
    x_train: pd.DataFrame, y_train: pd.Series
) -> Dict[str, GridSearchCV]:
    """Perform grid search for each model."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    pipelines = create_pipelines()
    param_grids = create_param_grids()

    grid_search_results = {}
    for name, pipeline in pipelines.items():
        with mlflow.start_run(nested=True):
            grid_search = GridSearchCV(pipeline, param_grids[name], cv=cv)
            grid_search.fit(x_train, y_train)
            grid_search_results[name] = grid_search

            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric(f"{name}_best_score", grid_search.best_score_)

    return grid_search_results


def create_ensemble_models(
    grid_search_results: Dict[str, GridSearchCV]
) -> Tuple[VotingClassifier, VotingClassifier]:
    best_models = [
        grid_search.best_estimator_ for grid_search in grid_search_results.values()
    ]

    ensemble_soft = VotingClassifier(
        estimators=[
            (name, model)
            for name, model in zip(grid_search_results.keys(), best_models)
        ],
        voting="soft",
    )

    ensemble_hard = VotingClassifier(
        estimators=[
            (name, model)
            for name, model in zip(grid_search_results.keys(), best_models)
        ],
        voting="hard",
    )

    return ensemble_soft, ensemble_hard


def create_stacking_models(
    grid_search_results: Dict[str, GridSearchCV]
) -> Tuple[StackingClassifier, StackingClassifier]:
    best_models = [
        grid_search.best_estimator_ for grid_search in grid_search_results.values()
    ]

    stacking_logist = StackingClassifier(
        classifiers=best_models, meta_classifier=LogisticRegression()
    )

    stacking_lgbm = StackingClassifier(
        classifiers=best_models, meta_classifier=LGBMClassifier()
    )

    return stacking_logist, stacking_lgbm


def evaluate_model(
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[float, float]:
    try:
        print(f"Shape of x_train in evaluate_model: {x_train.shape}")
        print(f"Shape of y_train in evaluate_model: {y_train.shape}")
        print(f"Shape of x_test in evaluate_model: {x_test.shape}")
        print(f"Shape of y_test in evaluate_model: {y_test.shape}")
        print("Evaluating model...{model.__class__.__name__}")
        # Ensure y is 1d
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()

        print(f"Shape of y_train after squeeze: {y_train.shape}")
        print(f"Shape of y_test after squeeze: {y_test.shape}")

        if isinstance(model, (VotingClassifier, StackingClassifier)):
            # For ensemble models, we need to fit them explicitly
            model.fit(x_train, y_train)

        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)

        print(f"Shape of train_pred: {train_pred.shape}")
        print(f"Shape of test_pred: {test_pred.shape}")

        train_score = round(accuracy_score(y_train, train_pred), 3)
        test_score = round(accuracy_score(y_test, test_pred), 3)

        report = classification_report(y_test, test_pred, output_dict=True)
        mlflow.log_metrics(
            {
                f"{model.__class__.__name__}_precision": report["weighted avg"][
                    "precision"
                ],
                f"{model.__class__.__name__}_recall": report["weighted avg"]["recall"],
                f"{model.__class__.__name__}_f1-score": report["weighted avg"][
                    "f1-score"
                ],
            }
        )

        cm = confusion_matrix(y_test, test_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"Confusion Matrix - {model.__class__.__name__}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.savefig(f"{MODEL_DIR}/{model.__class__.__name__}_confusion_matrix.png")
        mlflow.log_artifact(
            f"{MODEL_DIR}/{model.__class__.__name__}_confusion_matrix.png"
        )
        plt.close()

        return train_score, test_score
    except Exception as e:
        print(f"Error evaluating model {model.__class__.__name__}: {str(e)}")
        return 0.0, 0.0


def save_models_and_results(
    models: List,
    names: List[str],
    grid_search_results: Dict[str, GridSearchCV],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict]:
    os.makedirs(MODEL_DIR, exist_ok=True)
    results = {}
    for name, model in zip(names, models):
        joblib.dump(model, f"{MODEL_DIR}/{name}_model.joblib")
        print(f"evaluating model {name}...{model.__class__.__name__}")

        train_score, test_score = evaluate_model(
            model, x_train, y_train, x_test, y_test
        )
        cv_score = grid_search_results.get(name, {}).get("best_score_", 0.0)

        results[name] = {
            "cv_score": cv_score,
            "train_score": train_score,
            "test_score": test_score,
        }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f)

    joblib.dump((x_train, x_test, y_train, y_test), DATA_SPLITS_FILE)

    return results


def load_models_and_results(
    names: List[str],
) -> Tuple[List, Dict, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    models = [joblib.load(f"{MODEL_DIR}/{name}_model.joblib") for name in names]

    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)

    x_train, x_test, y_train, y_test = joblib.load(DATA_SPLITS_FILE)

    return models, results, x_train, x_test, y_train, y_test


def train_models(
    data_dir: str,
) -> Tuple[
    List,
    List[str],
    Dict[str, GridSearchCV],
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    x_train, x_test, y_train, y_test = process_data(data_dir)

    print(f"Shape of x_train: {x_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of x_test: {x_test.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    # Ensure y is 1D
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    print(f"Shape of y_train after squeeze: {y_train.shape}")
    print(f"Shape of y_test after squeeze: {y_test.shape}")

    if y_train.ndim > 1 or y_test.ndim > 1:
        raise ValueError(
            f"Target variable has incorrect shape. y_train: {y_train.shape}, y_test: {y_test.shape}"
        )

    grid_search_results = perform_grid_search(x_train, y_train)
    print("Grid search completed..................", grid_search_results)
    print(f"Shape of x_train: {x_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of x_test: {x_test.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    ensemble_soft, ensemble_hard = create_ensemble_models(grid_search_results)
    stacking_logist, stacking_lgbm = create_stacking_models(grid_search_results)

    models = [ensemble_soft, ensemble_hard, stacking_logist, stacking_lgbm]
    names = ["Ensemble_Soft", "Ensemble_Hard", "Stacking_Logistic", "Stacking_LGBM"]

    save_models_and_results(
        models, names, grid_search_results, x_train, y_train, x_test, y_test
    )

    return models, names, grid_search_results, x_train, x_test, y_train, y_test


def create_performance_plot(
    models: List,
    names: List[str],
    results: Dict[str, Dict],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    plt.figure(figsize=(12, 8))
    scores = pd.DataFrame(
        {
            "Model": names,
            "CV Score": [results[name]["cv_score"] for name in names],
            "Test Score": [results[name]["test_score"] for name in names],
        }
    )
    scores = scores.melt(id_vars="Model", var_name="Type", value_name="Score")
    sns.barplot(x="Model", y="Score", hue="Type", data=scores)
    plt.title("Model Performance Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/model_comparison.png")
    mlflow.log_artifact(f"{MODEL_DIR}/model_comparison.png")


def main(data_dir: str, rerun_evaluation: bool = False) -> None:
    client, experiment_id = setup_mlflow("Heart Disease Classification")

    names = ["Ensemble_Soft", "Ensemble_Hard", "Stacking_Logistic", "Stacking_LGBM"]

    with mlflow.start_run(experiment_id=experiment_id) as run:
        if not rerun_evaluation:
            print("Training models...")
            models, names, grid_search_results, x_train, x_test, y_train, y_test = (
                train_models(data_dir)
            )
            results = save_models_and_results(
                models, names, grid_search_results, x_train, y_train, x_test, y_test
            )
        else:
            print("Loading models...")
            models, results, x_train, x_test, y_train, y_test = load_models_and_results(
                names
            )
            print(
                "Loading models...models, results, x_train, x_test, y_train, y_test",
            )

        for name in names:
            print(f"Logging results for {name} model...")
            cv_score = results[name]["cv_score"]
            train_score = results[name]["train_score"]
            test_score = results[name]["test_score"]
            mlflow.log_metric(f"{name}_cv_score", float(cv_score) if cv_score else 0.0)
            mlflow.log_metric(f"{name}_train_score", float(train_score))
            mlflow.log_metric(f"{name}_test_score", float(test_score))
            mlflow.sklearn.log_model(models[names.index(name)], name)

        create_performance_plot(
            models, names, results, x_train, x_test, y_train, y_test
        )


def predict(model_name: str, data: pd.DataFrame) -> np.ndarray:
    """Make predictions using a saved model."""
    model = joblib.load(f"{MODEL_DIR}/{model_name}_model.joblib")
    return model.predict(data)


if __name__ == "__main__":
    main("./data/dataset_heart.csv", rerun_evaluation=False)
