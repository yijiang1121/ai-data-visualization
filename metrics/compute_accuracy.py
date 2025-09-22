"""Utilities for computing grading accuracy metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

REQUIRED_COLUMNS = {"Prompt_ID", "Human_Grade", "Auto_Grade", "Ground_Truth"}
DERIVED_GNP_COLUMNS = {
    "Auto_Grade_GNP": "Auto_Grade",
    "Human_Grade_GNP": "Human_Grade",
    "Ground_Truth_GNP": "Ground_Truth",
}

GNP_MAPPING = {
    "highly satisfying": "G",
    "slightly satisfying": "G",
    "slightly unsatisfying": "N",
    "highly unsatisfying": "P",
}

SCALE_CONFIGS = {
    "four_level": {
        "label": "4-Point Detail",
        "columns": {
            "ground_truth": "Ground_Truth",
            "autograder": "Auto_Grade",
            "human": "Human_Grade",
        },
    },
    "gnp": {
        "label": "G / N / P",
        "columns": {
            "ground_truth": "Ground_Truth_GNP",
            "autograder": "Auto_Grade_GNP",
            "human": "Human_Grade_GNP",
        },
    },
}

DEFAULT_SCALE_KEY = "four_level"

LABEL_COLUMNS = {
    column
    for config in SCALE_CONFIGS.values()
    for column in config["columns"].values()
}


def _normalize(series: pd.Series) -> pd.Series:
    """Normalize label text for reliable comparisons."""
    return series.astype(str).str.strip().str.lower()


def derive_gnp_label(value: object) -> str | None:
    """Return the ``G``/``N``/``P`` bucket for a detailed satisfaction label."""

    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    return GNP_MAPPING.get(text.lower())


def ensure_gnp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Populate the derived G/N/P columns based on the detailed ratings."""

    for derived_column, source_column in DERIVED_GNP_COLUMNS.items():
        if source_column not in df.columns:
            continue
        df[derived_column] = df[source_column].map(derive_gnp_label)

    return df


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

    return ensure_gnp_columns(df)


def _format(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _format_rate(numerator: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return _format(numerator / denominator)


def _safe_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = series.mean()
    return None if pd.isna(value) else float(value)


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


def _compute_accuracy_for_scale(
    normalized: pd.DataFrame, config: Dict[str, Dict[str, str]]
) -> Dict[str, object]:
    columns = config.get("columns", {})
    autograder_column = columns.get("autograder")
    human_column = columns.get("human")
    ground_truth_column = columns.get("ground_truth")

    if not all(
        column in normalized.columns
        for column in (autograder_column, human_column, ground_truth_column)
    ):
        return {"summary": {}, "per_prompt": []}

    autograder_matches = normalized[autograder_column] == normalized[ground_truth_column]
    human_matches = normalized[human_column] == normalized[ground_truth_column]

    total_evaluations = int(len(normalized))
    if "Prompt_ID" in normalized.columns:
        unique_prompts = int(normalized["Prompt_ID"].nunique())
    else:
        unique_prompts = total_evaluations

    if "Prompt_ID" in normalized.columns and unique_prompts:
        prompt_groups = normalized.groupby("Prompt_ID", sort=False)
        prompt_autograder = prompt_groups[autograder_column].first()
        prompt_ground_truth = prompt_groups[ground_truth_column].first()
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
            prompt_autograder = _safe_mean(
                group[autograder_column] == group[ground_truth_column]
            )
            prompt_human = _safe_mean(group[human_column] == group[ground_truth_column])
            per_prompt.append(
                {
                    "prompt_id": str(prompt_id),
                    "count": int(len(group)),
                    "autograder_accuracy": _format(prompt_autograder),
                    "human_accuracy": _format(prompt_human),
                }
            )

    return {"summary": summary, "per_prompt": per_prompt}


def compute_metrics(df: pd.DataFrame) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Compute accuracy and revision-focused metrics for every supported scale."""

    df = ensure_gnp_columns(df)
    normalized = df.copy()

    display_maps = {
        column: _build_display_map(df[column]) for column in LABEL_COLUMNS if column in df.columns
    }
    for column in LABEL_COLUMNS:
        if column in normalized.columns:
            normalized[column] = _normalize(normalized[column])

    accuracy_scales: Dict[str, Dict[str, object]] = {}
    for scale_key, config in SCALE_CONFIGS.items():
        accuracy_scales[scale_key] = _compute_accuracy_for_scale(normalized, config)

    default_accuracy = accuracy_scales.get(
        DEFAULT_SCALE_KEY,
        next(iter(accuracy_scales.values()), {"summary": {}, "per_prompt": []}),
    )

    revision_scales: Dict[str, Dict[str, object]] = {}
    for scale_key, config in SCALE_CONFIGS.items():
        revision_scales[scale_key] = _compute_revision_metrics_for_scale(
            normalized=normalized,
            display_maps=display_maps,
            config=config,
        )

    default_revision = revision_scales.get(
        DEFAULT_SCALE_KEY, next(iter(revision_scales.values()), {})
    )

    scale_labels = {key: value["label"] for key, value in SCALE_CONFIGS.items()}

    accuracy_metrics = {
        "default_scale": DEFAULT_SCALE_KEY,
        "scale_labels": scale_labels,
        "scales": accuracy_scales,
        "summary": default_accuracy.get("summary", {}),
        "per_prompt": default_accuracy.get("per_prompt", []),
    }

    revision_metrics = {
        "default_scale": DEFAULT_SCALE_KEY,
        "scale_labels": scale_labels,
        "scales": revision_scales,
    }
    revision_metrics.update(default_revision)

    return accuracy_metrics, revision_metrics


def _compute_revision_metrics_for_scale(
    *,
    normalized: pd.DataFrame,
    display_maps: Dict[str, Dict[str, str]],
    config: Dict[str, Dict[str, str]],
) -> Dict[str, object]:
    """Compute revision-focused metrics for a single grading scale."""

    columns = config.get("columns", {})
    autograder_column = columns.get("autograder")
    human_column = columns.get("human")
    ground_truth_column = columns.get("ground_truth")

    required_columns = [autograder_column, human_column, ground_truth_column]
    if not all(column in normalized.columns for column in required_columns):
        return {
            "overall": {},
            "cases": {},
            "autograder_wrong_breakdown": {},
            "breakdowns": {},
        }

    autograder_matches = normalized[autograder_column] == normalized[ground_truth_column]
    human_matches = normalized[human_column] == normalized[ground_truth_column]
    revisions = normalized[human_column] != normalized[autograder_column]

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
    corrected_revision_count = case_counts.get("autograder_wrong_human_correct", 0)
    both_wrong_count = case_counts.get("both_wrong", 0)
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
                case_counts.get("autograder_wrong_human_correct", 0),
                include_autograder_share=True,
            ),
            "autograder_correct_human_wrong": _case_entry(
                case_counts.get("autograder_correct_human_wrong", 0),
                include_autograder_share=False,
            ),
            "both_wrong": _case_entry(
                case_counts.get("both_wrong", 0), include_autograder_share=True
            ),
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
        ("ground_truth", ground_truth_column),
        ("autograder_label", autograder_column),
        ("human_label", human_column),
    )

    for dimension_key, column_name in dimension_columns:
        if not column_name or column_name not in normalized.columns:
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
            label_corrected = label_case_counts.get("autograder_wrong_human_correct", 0)
            label_both_wrong = label_case_counts.get("both_wrong", 0)
            label_autograder_wrong_revised = label_corrected + label_both_wrong
            label_autograder_wrong_unrevised = max(
                0, label_autograder_wrong_total - label_autograder_wrong_revised
            )

            def _dimension_case_entry(
                case_key: str, *, include_autograder_share: bool
            ) -> Dict[str, object]:
                count_value = label_case_counts.get(case_key, 0)
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
