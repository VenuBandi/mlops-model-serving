"""Train a demo XGBoost model and log everything to MLflow."""
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from xgboost import XGBClassifier

EXPERIMENT = "cdc-places-demo"
mlflow.set_experiment(EXPERIMENT)


def generate_demo_data(n: int = 10_000):
    X, y = make_classification(
        n_samples=n, n_features=20, n_informative=12,
        n_redundant=4, random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name="target")


def train():
    X, y = generate_demo_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
    }

    with mlflow.start_run(run_name="xgb_baseline") as run:
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        f1  = f1_score(y_test, preds)

        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("f1", f1)
        mlflow.xgboost.log_model(model, artifact_path="model",
                                  registered_model_name="xgb_demo")

        print(f"Run ID : {run.info.run_id}")
        print(f"ROC-AUC: {auc:.4f}  |  F1: {f1:.4f}")
        print(classification_report(y_test, preds))

    return run.info.run_id


if __name__ == "__main__":
    run_id = train()
    print(f"\nTo serve: mlflow models serve -m runs:/{run_id}/model -p 8001")
