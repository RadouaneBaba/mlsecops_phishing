# MLSecOps — Phishing Detection Pipeline

A secure MLOps pipeline for detecting malicious PDF and HTML attachments using XGBoost, with DevSecOps controls at every stage.

## Dataset

CIC-Trap4Phish 2025 — PDF and HTML formats only.

Place the CSV files in `data/` before running:
- `data/HTML_Top13_Features.csv`
- `data/PDF_Top10_features.csv`

## Setup

```bash
pip install -r requirements.txt
```

### DagHub configuration

1. Create a free account at [dagshub.com](https://dagshub.com)
2. Create a repository named `mlsecops-phishing`
3. Update `config/config.yaml` with your username
4. Add `DAGSHUB_TOKEN` as a GitHub Actions secret (Settings → Secrets → Actions)

## Running locally

```bash
# Validate data
python src/data_validation.py --dataset-type html
python src/data_validation.py --dataset-type pdf

# Train
python src/train.py --dataset-type html
python src/train.py --dataset-type pdf

# Evaluate
python src/evaluate.py --dataset-type html
python src/evaluate.py --dataset-type pdf

# Register
python src/register.py --dataset-type html
python src/register.py --dataset-type pdf
```

## Pipeline (GitHub Actions)

Every push to `main` triggers the full pipeline:

```
Security Scan (Bandit + Safety)
        ↓
Data Validation (HTML + PDF)
        ↓
Train HTML ──┬── Train PDF
             ↓           ↓
    Evaluate HTML   Evaluate PDF
             ↓           ↓
         Register (both models)
```

Any stage failure blocks all downstream stages.

## Project structure

```
├── .github/workflows/pipeline.yml   ← CI/CD
├── config/config.yaml               ← all settings, no hardcoded values
├── src/
│   ├── config_loader.py             ← shared config reader
│   ├── data_validation.py           ← Great Expectations checks
│   ├── train.py                     ← XGBoost training + MLflow logging
│   ├── evaluate.py                  ← per-class metrics + CI gate
│   └── register.py                  ← SHA-256 signing + model registry
├── data/                            ← gitignored, add CSVs here
├── models/                          ← gitignored, populated by train.py
├── hashes/                          ← gitignored, SHA-256 files
├── reports/                         ← gitignored, evaluation JSONs
└── requirements.txt
```
