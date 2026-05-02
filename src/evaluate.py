"""
evaluate.py
───────────
Load a trained model and evaluate it with per-class metrics.
Fails (exit 1) if any metric drops below the threshold defined in config.yaml.

Usage:
    python src/evaluate.py --dataset-type html|pdf
                           [--data path/to/override.csv]
                           [--model path/to/override.pkl]

Exit codes:
    0  all metrics above threshold
    1  one or more metrics below threshold (blocks CI deployment)
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import mlflow
import dagshub

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import load_config


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(dataset_type: str, data_path: str, model_path: str, cfg: dict) -> bool:
    t_cfg   = cfg["training"]
    th_cfg  = cfg["thresholds"]
    m_cfg   = cfg["mlflow"]
    rep_cfg = cfg["reports"]

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Evaluation  |  type={dataset_type}")
    print(f"{'─'*60}")

    if not Path(model_path).exists():
        print(f"[ERROR] Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"  Model loaded  ← {model_path}")

    # ── Load and split data (same seed as training for reproducibility) ───────
    df = pd.read_csv(data_path)
    X  = df.drop(columns=["label", "file_name"], errors="ignore")
    y  = df["label"]

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=t_cfg["test_size"],
        stratify=y,
        random_state=t_cfg["random_state"],
    )
    print(f"  Test set: {X_test.shape[0]} rows\n")

    # ── Predictions ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc          = accuracy_score(y_test, y_pred)
    f1_weighted  = f1_score(y_test, y_pred, average="weighted")
    f1_benign    = f1_score(y_test, y_pred, pos_label=0, average="binary")
    f1_malicious = f1_score(y_test, y_pred, pos_label=1, average="binary")
    prec_benign    = precision_score(y_test, y_pred, pos_label=0, average="binary")
    prec_malicious = precision_score(y_test, y_pred, pos_label=1, average="binary")
    rec_benign     = recall_score(y_test, y_pred, pos_label=0, average="binary")
    rec_malicious  = recall_score(y_test, y_pred, pos_label=1, average="binary")
    cm             = confusion_matrix(y_test, y_pred).tolist()

    print("  Per-class metrics:")
    print(f"    Class 0 (benign)    — F1: {f1_benign:.4f}  P: {prec_benign:.4f}  R: {rec_benign:.4f}")
    print(f"    Class 1 (malicious) — F1: {f1_malicious:.4f}  P: {prec_malicious:.4f}  R: {rec_malicious:.4f}")
    print(f"    Weighted F1         : {f1_weighted:.4f}")
    print(f"    Accuracy            : {acc:.4f}")
    print(f"\n  Confusion matrix (TN FP / FN TP):")
    print(f"    {cm[0]}")
    print(f"    {cm[1]}")

    # ── Threshold gates ───────────────────────────────────────────────────────
    gates = {
        "f1_malicious":  (f1_malicious,  th_cfg["f1_malicious"]),
        "f1_benign":     (f1_benign,     th_cfg["f1_benign"]),
        "f1_weighted":   (f1_weighted,   th_cfg["f1_weighted"]),
        "accuracy":      (acc,           th_cfg["accuracy"]),
    }

    print(f"\n  Threshold gates:")
    all_passed = True
    gate_results = {}
    for metric, (value, threshold) in gates.items():
        passed = value >= threshold
        gate_results[metric] = {"value": round(value, 4), "threshold": threshold, "passed": passed}
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"    {status}  {metric}: {value:.4f} (min={threshold})")
        if not passed:
            all_passed = False

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    dagshub.init(
        repo_owner=m_cfg["dagshub_username"],
        repo_name=m_cfg["dagshub_repo"],
        mlflow=True,
    )
    mlflow.set_experiment(f"{m_cfg['experiment_prefix']}-{dataset_type}")

    with mlflow.start_run(run_name=f"eval-{dataset_type}"):
        mlflow.log_metric("accuracy",          acc)
        mlflow.log_metric("f1_weighted",       f1_weighted)
        mlflow.log_metric("f1_benign",         f1_benign)
        mlflow.log_metric("f1_malicious",      f1_malicious)
        mlflow.log_metric("precision_benign",  prec_benign)
        mlflow.log_metric("precision_malicious", prec_malicious)
        mlflow.log_metric("recall_benign",     rec_benign)
        mlflow.log_metric("recall_malicious",  rec_malicious)
        mlflow.log_param("dataset_type",       dataset_type)
        mlflow.log_param("evaluation_gate",    "passed" if all_passed else "failed")

    # ── Save report JSON ──────────────────────────────────────────────────────
    report = {
        "dataset_type": dataset_type,
        "model_path":   model_path,
        "metrics": {
            "accuracy":            round(acc, 4),
            "f1_weighted":         round(f1_weighted, 4),
            "f1_benign":           round(f1_benign, 4),
            "f1_malicious":        round(f1_malicious, 4),
            "precision_benign":    round(prec_benign, 4),
            "precision_malicious": round(prec_malicious, 4),
            "recall_benign":       round(rec_benign, 4),
            "recall_malicious":    round(rec_malicious, 4),
        },
        "confusion_matrix": cm,
        "gate_results":     gate_results,
        "all_passed":       all_passed,
    }

    report_path = Path(rep_cfg[dataset_type])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved → {report_path}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    if all_passed:
        print("  ✅  EVALUATION PASSED — model cleared for registration")
    else:
        print("  ❌  EVALUATION FAILED — deployment blocked")
    print(f"{'─'*60}\n")

    return all_passed


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Evaluate trained XGBoost model.")
    parser.add_argument("--dataset-type", required=True, choices=["html", "pdf"], dest="dataset_type")
    parser.add_argument("--data",  default=None, help="Override data path from config")
    parser.add_argument("--model", default=None, help="Override model path from config")
    args = parser.parse_args()

    data_path  = args.data  if args.data  else cfg["data"][args.dataset_type]
    model_path = args.model if args.model else cfg["model"][args.dataset_type]

    ok = evaluate(args.dataset_type, data_path, model_path, cfg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
