"""Utilities for computing grading accuracy metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

CASE_KEYS = (
    "autograder_wrong_human_correct",
    "autograder_correct_human_wrong",
    "both_wrong",
)

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


def _format(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _format_rate(numerator: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return _format(numerator / denominator)


def _build_display_map(series: pd.Series) -> Dict[str, str]:
    """Map normalized label values back to their display text."""

    cleaned = series.dropna()
    if cleaned.empty:
        return {}

    normalized_col = cleaned.astype(str).str.strip().str.lower()
    display_col = cleaned.astype(str).str.strip()
    map_df = (
        pd.DataFrame({"normalized": normalized_col, "display": display_col})
        .drop_duplicates(subset="normalized")
        .set_index("normalized")
    )
    return map_df["display"].to_dict()


def compute_metrics(df: pd.DataFrame) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Compute accuracy and revision-focused metrics."""

    normalized = df.copy()
    display_maps = {
        column: _build_display_map(df[column]) for column in LABEL_COLUMNS if column in df.columns
    }
    for column in LABEL_COLUMNS:
        normalized[column] = _normalize(normalized[column])

    autograder_matches = normalized["Auto_Grade"] == normalized["Ground_Truth"]
    human_matches = normalized["Human_Grade"] == normalized["Ground_Truth"]
    revisions = normalized["Human_Grade"] != normalized["Auto_Grade"]

    def _safe_mean(series: pd.Series) -> float | None:
        if series.empty:
            return None
        value = series.mean()
        return None if pd.isna(value) else float(value)

    total_evaluations = int(len(normalized))
    unique_prompts = (
        int(normalized["Prompt_ID"].nunique())
        if "Prompt_ID" in normalized.columns
        else total_evaluations
    )

    if "Prompt_ID" in normalized.columns and unique_prompts:
        prompt_groups = normalized.groupby("Prompt_ID", sort=False)
        prompt_autograder = prompt_groups["Auto_Grade"].first()
        prompt_ground_truth = prompt_groups["Ground_Truth"].first()
        autograder_accuracy_value = _safe_mean(prompt_autograder == prompt_ground_truth)
    else:
        autograder_accuracy_value = _safe_mean(autograder_matches)

    summary = {
        "total_evaluations": total_evaluations,
        "unique_prompts": unique_prompts,
        "autograder_accuracy": _format(autograder_accuracy_value),
        "autograder_evaluations": unique_prompts,
        "human_accuracy": _format(_safe_mean(human_matches)),
        "human_evaluations": total_evaluations,
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

    revision = compute_revision_metrics(
        normalized=normalized,
        display_maps=display_maps,
        autograder_matches=autograder_matches,
        human_matches=human_matches,
        revisions=revisions,
    )

    return {"summary": summary, "per_prompt": per_prompt}, revision


def compute_revision_metrics(
    *,
    normalized: pd.DataFrame,
    display_maps: Dict[str, Dict[str, str]],
    autograder_matches: pd.Series,
    human_matches: pd.Series,
    revisions: pd.Series,
) -> Dict[str, object]:
    """Compute revision-focused metrics including precision and recall."""

    total_evaluations = int(len(normalized))
    autograder_wrong_mask = ~autograder_matches

    case_masks = {
        "autograder_wrong_human_correct": autograder_wrong_mask & human_matches & revisions,
        "autograder_correct_human_wrong": autograder_matches & (~human_matches) & revisions,
        "both_wrong": autograder_wrong_mask & (~human_matches) & revisions,
    }

    case_counts = {case: int(mask.sum()) for case, mask in case_masks.items()}
    revision_count = int(revisions.sum())
    autograder_wrong_total = int(autograder_wrong_mask.sum())
    corrected_revision_count = case_counts["autograder_wrong_human_correct"]
    both_wrong_count = case_counts["both_wrong"]
    autograder_wrong_revised = corrected_revision_count + both_wrong_count
    autograder_wrong_unrevised = max(0, autograder_wrong_total - autograder_wrong_revised)

    def _case_entry(count: int, *, include_autograder_share: bool) -> Dict[str, object]:
        entry = {
            "count": count,
            "share_of_revisions": _format_rate(count, revision_count),
            "share_of_total": _format_rate(count, total_evaluations),
        }
        if include_autograder_share:
            entry["share_of_autograder_wrong"] = _format_rate(count, autograder_wrong_total)
        else:
            entry["share_of_autograder_wrong"] = None
        return entry

    def _breakdown_entry(count: int) -> Dict[str, object]:
        return {
            "count": count,
            "share_of_autograder_wrong": _format_rate(count, autograder_wrong_total),
            "share_of_total": _format_rate(count, total_evaluations),
        }

    unique_repetitions = 1
    if "Repetition" in normalized.columns:
        unique_values = normalized["Repetition"].dropna().unique()
        if len(unique_values):
            unique_repetitions = int(len(unique_values))

    revision = {
        "overall": {
            "total_evaluations": total_evaluations,
            "revision_count": revision_count,
            "revision_rate": _format_rate(revision_count, total_evaluations),
            "correct_revision_count": corrected_revision_count,
            "correct_revision_precision": _format_rate(
                corrected_revision_count, revision_count
            ),
            "autograder_wrong_total": autograder_wrong_total,
            "corrected_autograder_wrong": corrected_revision_count,
            "autograder_wrong_recall": _format_rate(
                corrected_revision_count, autograder_wrong_total
            ),
            "mistake_repetition_factor": unique_repetitions,
        },
        "cases": {
            "autograder_wrong_human_correct": _case_entry(
                case_counts["autograder_wrong_human_correct"], include_autograder_share=True
            ),
            "autograder_correct_human_wrong": _case_entry(
                case_counts["autograder_correct_human_wrong"], include_autograder_share=False
            ),
            "both_wrong": _case_entry(case_counts["both_wrong"], include_autograder_share=True),
        },
        "autograder_wrong_breakdown": {
            "corrected": _breakdown_entry(corrected_revision_count),
            "not_revised": _breakdown_entry(autograder_wrong_unrevised),
            "revised_but_wrong": _breakdown_entry(both_wrong_count),
        },
        "breakdowns": {},
    }

    def _resolve_display(column: str, value: str) -> str:
        mapping = display_maps.get(column, {})
        return mapping.get(value, value.title() if isinstance(value, str) else str(value))

    def _attach_dimension_fields(
        entry: Dict[str, object], dimension: str, label: str, display_label: str
    ) -> None:
        entry["label"] = label
        entry["label_display"] = display_label
        if dimension == "ground_truth":
            entry["ground_truth"] = label
            entry["ground_truth_display"] = display_label
        elif dimension == "autograder_label":
            entry["autograder_label"] = label
            entry["autograder_label_display"] = display_label
        elif dimension == "human_label":
            entry["human_label"] = label
            entry["human_label_display"] = display_label

    breakdowns: Dict[str, List[Dict[str, object]]] = {}
    dimension_columns = (
        ("ground_truth", "Ground_Truth"),
        ("autograder_label", "Auto_Grade"),
        ("human_label", "Human_Grade"),
    )

    for dimension_key, column_name in dimension_columns:
        if column_name not in normalized.columns:
            continue

        entries: List[Dict[str, object]] = []
        for label, _group in normalized.groupby(column_name, sort=True):
            if pd.isna(label):
                continue

            label_mask = normalized[column_name] == label
            label_total = int(label_mask.sum())
            if label_total == 0:
                continue

            label_revision_count = int(revisions[label_mask].sum())
            label_case_counts = {
                case: int(mask[label_mask].sum()) for case, mask in case_masks.items()
            }
            label_autograder_wrong_total = int((autograder_wrong_mask & label_mask).sum())
            label_corrected = label_case_counts["autograder_wrong_human_correct"]
            label_both_wrong = label_case_counts["both_wrong"]
            label_autograder_wrong_revised = label_corrected + label_both_wrong
            label_autograder_wrong_unrevised = max(
                0, label_autograder_wrong_total - label_autograder_wrong_revised
            )

            def _dimension_case_entry(
                case_key: str, *, include_autograder_share: bool
            ) -> Dict[str, object]:
                count_value = label_case_counts[case_key]
                entry_case = {
                    "count": count_value,
                    "share_of_revisions": _format_rate(count_value, label_revision_count),
                    "share_of_total": _format_rate(count_value, label_total),
                }
                if include_autograder_share:
                    entry_case["share_of_autograder_wrong"] = _format_rate(
                        count_value, label_autograder_wrong_total
                    )
                else:
                    entry_case["share_of_autograder_wrong"] = None
                return entry_case

            def _dimension_breakdown_entry(count_value: int) -> Dict[str, object]:
                return {
                    "count": count_value,
                    "share_of_autograder_wrong": _format_rate(
                        count_value, label_autograder_wrong_total
                    ),
                    "share_of_total": _format_rate(count_value, label_total),
                }

            display_label = _resolve_display(column_name, label)
            entry = {
                "total_evaluations": label_total,
                "revision_count": label_revision_count,
                "revision_rate": _format_rate(label_revision_count, label_total),
                "correct_revision_count": label_corrected,
                "correct_revision_precision": _format_rate(
                    label_corrected, label_revision_count
                ),
                "autograder_wrong_total": label_autograder_wrong_total,
                "corrected_autograder_wrong": label_corrected,
                "autograder_wrong_recall": _format_rate(
                    label_corrected, label_autograder_wrong_total
                ),
                "cases": {
                    "autograder_wrong_human_correct": _dimension_case_entry(
                        "autograder_wrong_human_correct", include_autograder_share=True
                    ),
                    "autograder_correct_human_wrong": _dimension_case_entry(
                        "autograder_correct_human_wrong", include_autograder_share=False
                    ),
                    "both_wrong": _dimension_case_entry(
                        "both_wrong", include_autograder_share=True
                    ),
                },
                "autograder_wrong_breakdown": {
                    "corrected": _dimension_breakdown_entry(label_corrected),
                    "not_revised": _dimension_breakdown_entry(
                        label_autograder_wrong_unrevised
                    ),
                    "revised_but_wrong": _dimension_breakdown_entry(label_both_wrong),
                },
            }
            _attach_dimension_fields(entry, dimension_key, label, display_label)
            entries.append(entry)

        if entries:
            breakdowns[dimension_key] = entries

    revision["breakdowns"] = breakdowns
    if "ground_truth" in breakdowns:
        revision["by_ground_truth"] = breakdowns["ground_truth"]

    return revision


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
    parser.add_argument(
        "--revision-output",
        type=Path,
        help=(
            "Path where revision metrics will be written. "
            "Defaults to 'revision.json' alongside the main output."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)
    accuracy_metrics, revision_metrics = compute_metrics(df)
    write_metrics(accuracy_metrics, args.output)
    revision_output = args.revision_output or args.output.with_name("revision.json")
    write_metrics(revision_metrics, revision_output)
    print(f"Wrote accuracy metrics to {args.output}")
    print(f"Wrote revision metrics to {revision_output}")


if __name__ == "__main__":
    main()
