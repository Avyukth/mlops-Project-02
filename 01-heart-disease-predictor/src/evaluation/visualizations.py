import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mlflow

def plot_feature_importance(model, feature_names, top_n=10):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = abs(model.coef_[0])
    else:
        raise ValueError("Model doesn't have feature_importances_ or coef_ attribute")

    feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_imp = feature_imp.sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_imp)
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    mlflow.log_artifact('feature_importance.png')

def plot_roc_curve(y_true, y_pred_proba, model_name):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    mlflow.log_artifact('roc_curve.png')

def create_performance_plot(results: dict, output_path: str):
    models = list(results.keys())
    train_scores = [results[model]['train_score'] for model in models]
    test_scores = [results[model]['test_score'] for model in models]

    plt.figure(figsize=(12, 6))
    x = range(len(models))
    width = 0.35

    plt.bar([i - width/2 for i in x], train_scores, width, label='Train Score', color='skyblue')
    plt.bar([i + width/2 for i in x], test_scores, width, label='Test Score', color='orange')

    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path)
    mlflow.log_artifact(output_path)
