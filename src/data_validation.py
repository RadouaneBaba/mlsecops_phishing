"""
data_validation.py
──────────────────
Validates a CSV dataset against a schema using Great Expectations 1.x.

Usage:
    python src/data_validation.py --dataset-type html|pdf
                                  [--data path/to/override.csv]
                                  [--output-hash path/to/hash.txt]

Exit codes:
    0  all checks passed
    1  one or more checks failed
"""

import sys
import argparse
import hashlib
from pathlib import Path

import pandas as pd
import great_expectations as gx
from great_expectations.expectations import (
    ExpectColumnToExist,
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToBeInSet,
    ExpectColumnValuesToNotBeNull,
)

# Load central config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import load_config


# ── Schema definitions ────────────────────────────────────────────────────────

HTML_SCHEMA = {
    "expected_columns": [
        "file_name",
        "url_punct_char_count",
        "tag_count",
        "whitespace_ratio",
        "entropy",
        "form_count",
        "embedded_js_count",
        "html_whitespace_ratio",
        "script_entropy",
        "min_link_length",
        "external_link_count",
        "total_script_characters",
        "internal_link_count",
        "url_digit_count",
        "label",
    ],
    "types": {
        "file_name": "object",
        "url_punct_char_count": "int64",
        "tag_count": "int64",
        "whitespace_ratio": "float64",
        "entropy": "float64",
        "form_count": "int64",
        "embedded_js_count": "int64",
        "html_whitespace_ratio": "float64",
        "script_entropy": "float64",
        "min_link_length": "int64",
        "external_link_count": "int64",
        "total_script_characters": "int64",
        "internal_link_count": "int64",
        "url_digit_count": "int64",
        "label": "int64",
    },
    "non_negative_int_cols": [
        "url_punct_char_count",
        "tag_count",
        "form_count",
        "embedded_js_count",
        "min_link_length",
        "external_link_count",
        "total_script_characters",
        "internal_link_count",
        "url_digit_count",
    ],
    "float_range_cols": {
        "whitespace_ratio": (0.0, 1.0),
        "entropy": (0.0, None),
        "html_whitespace_ratio": (0.0, 1.0),
        "script_entropy": (0.0, None),
    },
    "label_values": [0, 1],
    "not_null_cols": "__all__",
}

PDF_SCHEMA = {
    "expected_columns": [
        "text_length",
        "total_filters",
        "title_chars",
        "file_size",
        "object_count",
        "stream_count",
        "endstream_count",
        "metadata_size",
        "valid_pdf_header",
        "entropy_of_streams",
        "label",
    ],
    "types": {
        "text_length": "int64",
        "total_filters": "int64",
        "title_chars": "int64",
        "file_size": "int64",
        "object_count": "int64",
        "stream_count": "int64",
        "endstream_count": "int64",
        "metadata_size": "int64",
        "valid_pdf_header": "int64",
        "entropy_of_streams": "float64",
        "label": "int64",
    },
    "non_negative_int_cols": [
        "text_length",
        "total_filters",
        "title_chars",
        "file_size",
        "object_count",
        "stream_count",
        "endstream_count",
        "metadata_size",
    ],
    "float_range_cols": {
        "entropy_of_streams": (0.0, None),
    },
    "label_values": [0, 1],
    "not_null_cols": "__all__",
}

SCHEMAS = {"html": HTML_SCHEMA, "pdf": PDF_SCHEMA}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _record(
    results: list[dict], name: str, passed: bool, details: dict
) -> bool:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}")
    results.append({"check": name, "passed": passed, "details": details})
    return passed


def _check_name_from_ge_result(r) -> str:
    """
    Build a human-readable check name from ExpectationValidationResult.
    ExpectationValidationResult.expectation_config is an ExpectationConfiguration
    whose .type holds the snake_case expectation name and .kwargs holds parameters.
    """
    cfg = r.expectation_config
    exp_type: str = cfg.type
    kwargs: dict = cfg.kwargs

    col = kwargs.get("column", "")

    if exp_type == "expect_column_to_exist":
        return f"column_exists:{col}"
    if exp_type == "expect_column_values_to_not_be_null":
        return f"not_null:{col}"
    if exp_type == "expect_column_values_to_be_between":
        lo = kwargs.get("min_value")
        hi = kwargs.get("max_value")
        # Non-negative check: min=0, no max
        if lo == 0 and hi is None:
            return f"non_negative:{col}"
        return f"range:{col}[{lo},{hi}]"
    if exp_type == "expect_column_values_to_be_in_set":
        return "label_values_in_set"

    return f"{exp_type}:{col}" if col else exp_type


# ── Validation runner ─────────────────────────────────────────────────────────


def validate(df: pd.DataFrame, schema: dict) -> tuple[bool, list[dict]]:
    """
    Run all checks against *df* using the GE API.

    Returns (all_passed, results_list).
    """
    results: list[dict] = []
    all_passed = True
    existing_cols = set(df.columns)

    # ── 1. Direct pandas checks ──────────────────────────────────────────────
    # These three checks have no clean GE 1.x equivalent.

    # 1a. No extra columns
    extra = existing_cols - set(schema["expected_columns"])
    if not _record(
        results, "no_extra_columns", len(extra) == 0, {"extra": list(extra)}
    ):
        all_passed = False

    # 1b. Data type checks
    for col, expected_type in schema["types"].items():
        if col not in existing_cols:
            continue
        actual = str(df[col].dtype)
        passed = actual == expected_type
        if not _record(
            results,
            f"dtype:{col}",
            passed,
            {"expected": expected_type, "actual": actual},
        ):
            all_passed = False

    # 1c. Minimum row count
    row_ok = len(df) >= 10
    if not _record(results, "min_row_count", row_ok, {"rows": len(df)}):
        all_passed = False

    # ── 2. GE expectation suite ─────────────────────────────────────────
    # Build an ephemeral context (no filesystem project required), register the
    # dataframe as a one-shot batch, compile all expectations into a suite, and
    # run a ValidationDefinition.

    context = gx.get_context(mode="ephemeral")

    # Data source → asset → batch definition
    data_source = context.data_sources.add_pandas("pandas_source")
    data_asset = data_source.add_dataframe_asset("dataset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "full_batch"
    )

    # Collect expectations
    expectations: list = []

    # 2a. Column presence
    for col in schema["expected_columns"]:
        expectations.append(ExpectColumnToExist(column=col))

    # 2b. No-null checks
    null_cols = (
        schema["expected_columns"]
        if schema["not_null_cols"] == "__all__"
        else schema["not_null_cols"]
    )
    for col in null_cols:
        if col in existing_cols:
            expectations.append(ExpectColumnValuesToNotBeNull(column=col))

    # 2c. Non-negative integer columns
    for col in schema.get("non_negative_int_cols", []):
        if col in existing_cols:
            expectations.append(
                ExpectColumnValuesToBeBetween(column=col, min_value=0)
            )

    # 2d. Float range checks
    for col, (lo, hi) in schema.get("float_range_cols", {}).items():
        if col in existing_cols:
            expectations.append(
                ExpectColumnValuesToBeBetween(
                    column=col, min_value=lo, max_value=hi
                )
            )

    # 2e. Label value set
    if "label" in existing_cols:
        expectations.append(
            ExpectColumnValuesToBeInSet(
                column="label", value_set=schema["label_values"]
            )
        )

    # Register suite and validation definition
    suite = context.suites.add(
        gx.ExpectationSuite(name="validation_suite", expectations=expectations)
    )
    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="dataset_validation",
            data=batch_definition,
            suite=suite,
        )
    )

    # Run — pass the live dataframe via batch_parameters
    ge_result = validation_definition.run(batch_parameters={"dataframe": df})

    # Process GE results
    for r in ge_result.results:
        name = _check_name_from_ge_result(r)
        passed = bool(r.success)
        # r.result is a dict of observed values (unexpected_count, etc.)
        if not _record(results, name, passed, dict(r.result or {})):
            all_passed = False

    return all_passed, results


# ── Entry point ───────────────────────────────────────────────────────────────


def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser(description="Validate a CSV dataset.")
    p.add_argument(
        "--dataset-type",
        required=True,
        choices=["html", "pdf"],
        dest="dataset_type",
    )
    p.add_argument(
        "--data", default=None, help="Override data path from config"
    )
    p.add_argument(
        "--output-hash",
        default=None,
        dest="output_hash",
        help="Optional path to write the SHA-256 hash of the dataset",
    )
    args = p.parse_args()

    if args.data is None:
        args.data = cfg["data"][args.dataset_type]
    if args.output_hash is None:
        args.output_hash = cfg["hashes"][args.dataset_type].replace(
            ".sha256", "_data.sha256"
        )

    return args


def main():
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        print(f"[ERROR] File not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    schema = SCHEMAS[args.dataset_type]

    print(f"\n{'─'*60}")
    print(
        f"  Data Validation  |  type={args.dataset_type}  |  file={data_path.name}"
    )
    print(f"{'─'*60}")

    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns\n")

    sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    print(f"  Dataset SHA-256: {sha256}\n")

    if args.output_hash:
        Path(args.output_hash).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_hash).write_text(sha256)
        print(f"  Hash written to: {args.output_hash}\n")

    all_passed, results = validate(df, schema)

    n_total = len(results)
    n_passed = sum(1 for r in results if r["passed"])
    n_failed = n_total - n_passed

    print(f"\n{'─'*60}")
    if all_passed:
        print(f"  ✅  ALL {n_total} CHECKS PASSED")
    else:
        print(f"  ❌  {n_failed}/{n_total} CHECKS FAILED")
    print(f"{'─'*60}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
