"""
register.py
───────────
Compute SHA-256 hash of the trained model file and register it
in the MLflow Model Registry.

Usage:
    python src/register.py --dataset-type html|pdf
                           [--model path/to/override.pkl]

Exit codes:
    0  registration and signing succeeded
    1  any step failed
"""

import argparse
import hashlib
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient
import dagshub

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import load_config


# ── Hash ──────────────────────────────────────────────────────────────────────


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Registration ──────────────────────────────────────────────────────────────


def register(dataset_type: str, model_path: str, cfg: dict) -> bool:
    m_cfg = cfg["mlflow"]
    r_cfg = cfg["registry"]
    h_cfg = cfg["hashes"]

    print(f"\n{'─'*60}")
    print(f"  Model Registration  |  type={dataset_type}")
    print(f"{'─'*60}")

    if not Path(model_path).exists():
        print(f"[ERROR] Model not found: {model_path}", file=sys.stderr)
        return False

    # ── Step 1: Compute and save SHA-256 ──────────────────────────────────────
    sha256 = compute_sha256(model_path)
    print(f"\n  SHA-256: {sha256}")

    hash_path = Path(h_cfg[dataset_type])
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(sha256)
    print(f"  Hash saved → {hash_path}")

    # ── Step 2: Connect to DagHub / MLflow ────────────────────────────────────
    dagshub.init(
        repo_owner=m_cfg["dagshub_username"],
        repo_name=m_cfg["dagshub_repo"],
        mlflow=True,
    )

    client = MlflowClient()
    model_name = r_cfg[f"model_name_{dataset_type}"]

    # ── Step 3: Find latest run ───────────────────────────────────────────────
    experiment = mlflow.get_experiment_by_name(
        f"{m_cfg['experiment_prefix']}-{dataset_type}"
    )
    if experiment is None:
        print(
            f"[ERROR] Experiment not found. Run train.py first.",
            file=sys.stderr,
        )
        return False

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string='tags.mlflow.runName LIKE "xgb-%"',
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        print(
            "[ERROR] No training run found. Run train.py first.",
            file=sys.stderr,
        )
        return False

    latest_run = runs[0]
    run_id = latest_run.info.run_id

    # ── Step 3b: MLflow 3.x — find the LoggedModel for this run ─────────────
    # In MLflow 3.x models are stored as LoggedModels, not run artifacts.
    # runs:/<run_id>/model no longer works; we need the LoggedModel ID.
    logged_models = client.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"source_run_id = '{run_id}'",
        max_results=1,
    )
    if not logged_models:
        print(
            "[ERROR] No LoggedModel found for this run. ",
            file=sys.stderr,
        )
        return False

    logged_model = logged_models[0]
    model_uri = f"models:/{logged_model.model_id}"

    print(f"\n  Registering from run: {run_id}")
    print(f"  LoggedModel ID : {logged_model.model_id}")
    print(f"  Model URI      : {model_uri}")
    print(f"  Registry name  : {model_name}")

    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = mv.version
    print(f"  Registered as version: {version}")

    # ── Step 4: Transition to configured stage ────────────────────────────────
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=r_cfg["stage"],
        archive_existing_versions=True,
    )
    print(f"  Stage set to: {r_cfg['stage']}")

    # ── Step 5: Tag with hash for traceability ────────────────────────────────
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="sha256",
        value=sha256,
    )
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="dataset_type",
        value=dataset_type,
    )
    print(f"  SHA-256 tag set on registry version")

    print(f"\n{'─'*60}")
    print(f"  ✅  REGISTRATION COMPLETE  |  {model_name} v{version}")
    print(f"{'─'*60}\n")

    return True


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(
        description="Register trained model in MLflow registry."
    )
    parser.add_argument(
        "--dataset-type",
        required=True,
        choices=["html", "pdf"],
        dest="dataset_type",
    )
    parser.add_argument(
        "--model", default=None, help="Override model path from config"
    )
    args = parser.parse_args()

    model_path = args.model if args.model else cfg["model"][args.dataset_type]

    ok = register(args.dataset_type, model_path, cfg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
