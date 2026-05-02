"""
train.py
────────
Train an XGBoost binary classifier and log everything to MLflow (DagHub).

Usage:
    python src/train.py --dataset-type html|pdf
                        [--data path/to/override.csv]

Exit codes:
    0  training succeeded
    1  training failed
"""

import argparse
import hashlib
import sys
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
import dagshub

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import load_config

# ── Helpers ───────────────────────────────────────────────────────────────────


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data(data_path: str, test_size: float, random_state: int):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["label", "file_name"], errors="ignore")
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, X, y


# ── Training ──────────────────────────────────────────────────────────────────


def train(dataset_type: str, data_path: str, cfg: dict) -> bool:
    t_cfg = cfg["training"]
    m_cfg = cfg["mlflow"]
    ml_cfg = cfg["model"]

    print(f"\nLoading data: {data_path}")
    data_sha256 = file_hash(data_path)
    print(f"Dataset SHA-256: {data_sha256}")

    X_train, X_test, y_train, y_test, X, y = load_data(
        data_path, t_cfg["test_size"], t_cfg["random_state"]
    )
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    params = {
        "n_estimators": t_cfg["n_estimators"],
        "random_state": t_cfg["random_state"],
        "eval_metric": t_cfg["eval_metric"],
    }

    # ── DagHub / MLflow setup ─────────────────────────────────────────────────

    os.environ["DAGSHUB_USER_TOKEN"] = os.environ.get("DAGSHUB_TOKEN", "")

    dagshub.init(
        repo_owner=m_cfg["dagshub_username"],
        repo_name=m_cfg["dagshub_repo"],
        mlflow=True,
    )
    experiment_name = f"{m_cfg['experiment_prefix']}-{dataset_type}"
    mlflow.set_experiment(experiment_name)

    # ── Model output path from config ─────────────────────────────────────────
    model_output = Path(ml_cfg[dataset_type])
    model_output.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"xgb-{dataset_type}"):

        mlflow.log_params(params)
        mlflow.log_param("dataset_name", os.path.basename(data_path))
        mlflow.log_param("dataset_type", dataset_type)
        mlflow.log_param("test_size", t_cfg["test_size"])
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("features", list(X_train.columns))

        print("\nTraining XGBoost ...")
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # ── Metrics ───────────────────────────────────────────────────────────
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="weighted")

        cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=t_cfg["random_state"]
        )
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

        print(f"\n  Test  Accuracy : {test_acc:.4f}")
        print(f"  Test  F1       : {test_f1:.4f}")
        print(
            f"  CV    Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("cv_accuracy", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std", cv_scores.std())

        # ── Save model ────────────────────────────────────────────────────────
        with open(model_output, "wb") as f:
            pickle.dump(model, f)
        print(f"\n  Model saved → {model_output}")

        model_info = mlflow.xgboost.log_model(model, name="model")
        print(f"  LoggedModel ID : {model_info.model_id}")

        # verify other run artifacts (the model itself is now a LoggedModel)
        client = mlflow.tracking.MlflowClient()
        run_id = mlflow.active_run().info.run_id
        artifacts = client.list_artifacts(run_id)
        print(f"  Run artifacts  : {[a.path for a in artifacts]}")
        mlflow.log_artifact(str(model_output), artifact_path="model_file")

        print(f"  MLflow run ID  : {run_id}")

    print("\nTraining complete.\n")
    return True


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Train XGBoost classifier.")
    parser.add_argument(
        "--dataset-type",
        required=True,
        choices=["html", "pdf"],
        dest="dataset_type",
    )
    parser.add_argument(
        "--data", default=None, help="Override data path from config"
    )
    args = parser.parse_args()

    data_path = args.data if args.data else cfg["data"][args.dataset_type]

    ok = train(args.dataset_type, data_path, cfg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
