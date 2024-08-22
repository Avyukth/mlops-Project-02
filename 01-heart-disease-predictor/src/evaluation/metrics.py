import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)


def evaluate_model(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    """
    Evaluate the model and log metrics to MLflow.

    Args:
        model: The model to evaluate
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels

    Returns:
        tuple: A tuple containing (train_accuracy, test_accuracy)
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Log additional metrics
    test_report = classification_report(y_test, y_test_pred, output_dict=True)
    mlflow.log_metric("precision", test_report['weighted avg']['precision'])
    mlflow.log_metric("recall", test_report['weighted avg']['recall'])
    mlflow.log_metric("f1_score", test_report['weighted avg']['f1-score'])

    # Create and log confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    plt.close()
    mlflow.sklearn.log_model(model, f"{model.__class__.__name__}")
    mlflow.log_metric("train_score", train_accuracy)
    mlflow.log_metric("test_score", test_accuracy)
    return train_accuracy, test_accuracy

def log_feature_importance(model, feature_names):
    """
    Log feature importance if the model supports it.

    Args:
        model: The trained model
        feature_names (list): List of feature names
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return

    feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_imp = feature_imp.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_imp)
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    mlflow.log_artifact('feature_importance.png')
    plt.close()

    mlflow.log_table(feature_imp, "feature_importances.json")
