"""Utilities for computing grading accuracy metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

REQUIRED_COLUMNS = {"Prompt_ID", "Human_Grade", "Auto_Grade", "Ground_Truth"}
LABEL_COLUMNS = {"Human_Grade", "Auto_Grade", "Ground_Truth"}


def _normalize(series: pd.Series) -> pd.Series:
    """Normalize label text for reliable comparisons."""
    return series.astype(str).str.strip().str.lower()


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the grading dataset and validate required columns.

    Rows missing any required column are dropped to avoid skewed metrics.
    """

    df = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {csv_path}: {', '.join(sorted(missing_columns))}"
        )

    before_drop = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    if before_drop and len(df) < before_drop:
        print(
            f"Dropped {before_drop - len(df)} rows with incomplete data out of {before_drop} total"
        )

    return df


def compute_accuracy_metrics(df: pd.DataFrame) -> Dict[str, object]:
    """Compute overall and per-prompt accuracy metrics."""

    normalized = df.copy()
    for column in LABEL_COLUMNS:
        normalized[column] = _normalize(normalized[column])

    autograder_matches = normalized["Auto_Grade"] == normalized["Ground_Truth"]
    human_matches = normalized["Human_Grade"] == normalized["Ground_Truth"]

    def _safe_mean(series: pd.Series) -> float | None:
        if series.empty:
            return None
        value = series.mean()
        return None if pd.isna(value) else float(value)

    def _format(value: float | None) -> float | None:
        return None if value is None else round(value, 4)

    summary = {
        "total_records": int(len(normalized)),
        "autograder_accuracy": _format(_safe_mean(autograder_matches)),
        "human_accuracy": _format(_safe_mean(human_matches)),
    }

    per_prompt: List[Dict[str, object]] = []
    if "Prompt_ID" in normalized.columns:
        for prompt_id, group in normalized.groupby("Prompt_ID", sort=True):
            prompt_autograder = _safe_mean(group["Auto_Grade"] == group["Ground_Truth"])
            prompt_human = _safe_mean(group["Human_Grade"] == group["Ground_Truth"])
            per_prompt.append(
                {
                    "prompt_id": str(prompt_id),
                    "count": int(len(group)),
                    "autograder_accuracy": _format(prompt_autograder),
                    "human_accuracy": _format(prompt_human),
                }
            )

    return {"summary": summary, "per_prompt": per_prompt}


def write_metrics(metrics: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
        fh.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute grading accuracy metrics")
    parser.add_argument("--input", type=Path, required=True, help="Path to the grading dataset CSV")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the accuracy metrics JSON will be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)
    metrics = compute_accuracy_metrics(df)
    write_metrics(metrics, args.output)
    print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
