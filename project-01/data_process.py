from typing import Tuple

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(data_dir)
    mlflow.log_param("data_source", data_dir)
    mlflow.log_metric("num_rows", len(df))
    mlflow.log_metric("num_columns", len(df.columns))
    return df


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame()
    result["COLS"] = data.columns
    result["COUNT"] = data.count().values
    result["NULL"] = data.isna().sum().values
    result["NUNIQUE"] = data.nunique().values
    result["FREQ"] = data.mode().iloc[0].values
    result["MIN"] = data.describe().T["min"].values
    result["MAX"] = data.describe().T["max"].values
    result["MEAN"] = data.describe().T["mean"].values
    result["STD"] = data.describe().T["std"].values

    result.to_csv("data_summary.csv", index=False)
    mlflow.log_artifact("data_summary.csv")

    return result


def split_data_type(result: pd.DataFrame) -> Tuple[list, list]:
    cat_value = result[result["NUNIQUE"] < 5]["COLS"].tolist()
    num_value = result[result["NUNIQUE"] >= 5]["COLS"].tolist()

    mlflow.log_param("num_categorical_features", len(cat_value))
    mlflow.log_param("num_numerical_features", len(num_value))

    return cat_value, num_value


def preprocess_data(
    df: pd.DataFrame, target_column: str = "heart disease"
) -> Tuple[pd.DataFrame, pd.Series, list, list]:
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    summary = summarize(df)
    cat_value, num_value = split_data_type(summary)

    cat_value = [col for col in cat_value if col != target_column]

    for col in cat_value:
        X[col] = X[col].astype("category")

    X = pd.get_dummies(X)
    y = (y == 2).astype(int)  # Assuming 2 is the positive class, adjust if necessary

    mlflow.log_param("num_features_after_encoding", X.shape[1])
    mlflow.log_param("target_distribution", dict(y.value_counts(normalize=True)))

    return X, y, cat_value, num_value


def prepare_data(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2023
    )

    mlflow.log_param("train_size", len(x_train))
    mlflow.log_param("test_size", len(x_test))

    return x_train, x_test, y_train, y_test


def main(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    mlflow.set_experiment("Heart Disease Data Processing")

    # Check if there's an active run
    active_run = mlflow.active_run()
    if active_run:
        # If there's an active run, use it
        run = active_run
    else:
        # If there's no active run, start a new one
        run = mlflow.start_run()

    with run:
        # Read data
        df = read_data(data_dir)

        # Preprocess data
        X, y, cat_value, num_value = preprocess_data(df)

        # Prepare data for modeling
        x_train, x_test, y_train, y_test = prepare_data(X, y)

        # Log preprocessed data as artifacts
        X.to_csv("preprocessed_features.csv", index=False)
        y.to_csv("preprocessed_target.csv", index=False)
        mlflow.log_artifact("preprocessed_features.csv")
        mlflow.log_artifact("preprocessed_target.csv")

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    data_dir = "./data/dataset_heart.csv"  # Replace with your actual data path
    x_train, x_test, y_train, y_test = main(data_dir)
