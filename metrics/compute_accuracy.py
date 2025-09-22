"""Utilities for computing grading accuracy metrics."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ScaleColumns:
    """Convenience container for the expected columns of a grading scale."""

    ground_truth: str
    autograder: str
    human: str


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


def _resolve_scale_columns(
    df: pd.DataFrame, config: Dict[str, Dict[str, str]]
) -> ScaleColumns | None:
    """Return the configured column names if all are present in ``df``."""

    columns = config.get("columns", {})
    ground_truth = columns.get("ground_truth")
    autograder = columns.get("autograder")
    human = columns.get("human")

    if not all([ground_truth, autograder, human]):
        return None
    if not all(name in df.columns for name in (ground_truth, autograder, human)):
        return None

    return ScaleColumns(
        ground_truth=ground_truth, autograder=autograder, human=human
    )


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
    columns = _resolve_scale_columns(normalized, config)
    if not columns:
        return {"summary": {}, "per_prompt": []}

    autograder_matches = (
        normalized[columns.autograder] == normalized[columns.ground_truth]
    )
    human_matches = normalized[columns.human] == normalized[columns.ground_truth]

    total_evaluations = int(len(normalized))
    if "Prompt_ID" in normalized.columns:
        unique_prompts = int(normalized["Prompt_ID"].nunique())
    else:
        unique_prompts = total_evaluations

    if "Prompt_ID" in normalized.columns and unique_prompts:
        prompt_groups = normalized.groupby("Prompt_ID", sort=False)
        prompt_autograder = prompt_groups[columns.autograder].first()
        prompt_ground_truth = prompt_groups[columns.ground_truth].first()
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
                group[columns.autograder] == group[columns.ground_truth]
            )
            prompt_human = _safe_mean(
                group[columns.human] == group[columns.ground_truth]
            )
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


def _build_case_masks(
    *,
    autograder_matches: pd.Series,
    human_matches: pd.Series,
    revisions: pd.Series,
) -> Dict[str, pd.Series]:
    """Return boolean masks describing the different revision scenarios."""

    autograder_wrong = ~autograder_matches
    human_wrong = ~human_matches

    return {
        "autograder_wrong_human_correct": autograder_wrong & human_matches & revisions,
        "autograder_correct_human_wrong": autograder_matches & human_wrong & revisions,
        "both_wrong": autograder_wrong & human_wrong & revisions,
    }


def _case_stats(
    count: int,
    *,
    revision_count: int,
    total_evaluations: int,
    autograder_wrong_total: int,
    include_autograder_share: bool,
) -> Dict[str, object]:
    """Format the aggregate metrics for a single revision case."""

    entry = {
        "count": count,
        "share_of_revisions": _format_rate(count, revision_count),
        "share_of_total": _format_rate(count, total_evaluations),
    }
    if include_autograder_share:
        entry["share_of_autograder_wrong"] = _format_rate(
            count, autograder_wrong_total
        )
    else:
        entry["share_of_autograder_wrong"] = None
    return entry


def _breakdown_stats(
    count: int,
    *,
    total_evaluations: int,
    autograder_wrong_total: int,
) -> Dict[str, object]:
    """Format the breakdown metrics for autograder mistakes."""

    return {
        "count": count,
        "share_of_autograder_wrong": _format_rate(count, autograder_wrong_total),
        "share_of_total": _format_rate(count, total_evaluations),
    }


def _resolve_display_label(
    column: str, value: object, display_maps: Dict[str, Dict[str, str]]
) -> str:
    """Return a human-readable label for ``value`` in ``column``."""

    mapping = display_maps.get(column, {})
    if isinstance(value, str):
        return mapping.get(value, value.title())
    return mapping.get(value, str(value))


def _dimension_breakdowns(
    *,
    normalized: pd.DataFrame,
    column_name: str,
    dimension_key: str,
    display_maps: Dict[str, Dict[str, str]],
    case_masks: Dict[str, pd.Series],
    revisions: pd.Series,
    autograder_wrong_mask: pd.Series,
    total_evaluations: int,
) -> List[Dict[str, object]]:
    """Build revision breakdowns for a single dimension such as ground truth."""

    entries: List[Dict[str, object]] = []

    for label, group in normalized.groupby(column_name, sort=True):
        if pd.isna(label):
            continue

        label_total = int(len(group))
        if label_total == 0:
            continue

        indices = group.index
        label_revision_count = int(revisions.loc[indices].sum())
        label_case_counts = {
            case: int(mask.loc[indices].sum()) for case, mask in case_masks.items()
        }
        label_autograder_wrong_total = int(autograder_wrong_mask.loc[indices].sum())
        label_corrected = label_case_counts.get("autograder_wrong_human_correct", 0)
        label_both_wrong = label_case_counts.get("both_wrong", 0)
        label_autograder_wrong_revised = label_corrected + label_both_wrong
        label_autograder_wrong_unrevised = max(
            0, label_autograder_wrong_total - label_autograder_wrong_revised
        )

        display_label = _resolve_display_label(column_name, label, display_maps)

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
                "autograder_wrong_human_correct": _case_stats(
                    label_case_counts.get("autograder_wrong_human_correct", 0),
                    revision_count=label_revision_count,
                    total_evaluations=label_total,
                    autograder_wrong_total=label_autograder_wrong_total,
                    include_autograder_share=True,
                ),
                "autograder_correct_human_wrong": _case_stats(
                    label_case_counts.get("autograder_correct_human_wrong", 0),
                    revision_count=label_revision_count,
                    total_evaluations=label_total,
                    autograder_wrong_total=label_autograder_wrong_total,
                    include_autograder_share=False,
                ),
                "both_wrong": _case_stats(
                    label_case_counts.get("both_wrong", 0),
                    revision_count=label_revision_count,
                    total_evaluations=label_total,
                    autograder_wrong_total=label_autograder_wrong_total,
                    include_autograder_share=True,
                ),
            },
            "autograder_wrong_breakdown": {
                "corrected": _breakdown_stats(
                    label_corrected,
                    total_evaluations=label_total,
                    autograder_wrong_total=label_autograder_wrong_total,
                ),
                "not_revised": _breakdown_stats(
                    label_autograder_wrong_unrevised,
                    total_evaluations=label_total,
                    autograder_wrong_total=label_autograder_wrong_total,
                ),
                "revised_but_wrong": _breakdown_stats(
                    label_both_wrong,
                    total_evaluations=label_total,
                    autograder_wrong_total=label_autograder_wrong_total,
                ),
            },
            "label": label,
            "label_display": display_label,
        }

        if dimension_key == "ground_truth":
            entry["ground_truth"] = label
            entry["ground_truth_display"] = display_label
        elif dimension_key == "autograder_label":
            entry["autograder_label"] = label
            entry["autograder_label_display"] = display_label
        elif dimension_key == "human_label":
            entry["human_label"] = label
            entry["human_label_display"] = display_label

        entries.append(entry)

    return entries


def _compute_revision_metrics_for_scale(
    *,
    normalized: pd.DataFrame,
    display_maps: Dict[str, Dict[str, str]],
    config: Dict[str, Dict[str, str]],
) -> Dict[str, object]:
    """Compute revision-focused metrics for a single grading scale."""

    columns = _resolve_scale_columns(normalized, config)
    if not columns:
        return {
            "overall": {},
            "cases": {},
            "autograder_wrong_breakdown": {},
            "breakdowns": {},
        }

    autograder_matches = (
        normalized[columns.autograder] == normalized[columns.ground_truth]
    )
    human_matches = normalized[columns.human] == normalized[columns.ground_truth]
    revisions = normalized[columns.human] != normalized[columns.autograder]

    total_evaluations = int(len(normalized))
    autograder_wrong_mask = ~autograder_matches

    case_masks = _build_case_masks(
        autograder_matches=autograder_matches,
        human_matches=human_matches,
        revisions=revisions,
    )

    case_counts = {case: int(mask.sum()) for case, mask in case_masks.items()}
    revision_count = int(revisions.sum())
    autograder_wrong_total = int(autograder_wrong_mask.sum())
    corrected_revision_count = case_counts.get("autograder_wrong_human_correct", 0)
    both_wrong_count = case_counts.get("both_wrong", 0)
    autograder_wrong_revised = corrected_revision_count + both_wrong_count
    autograder_wrong_unrevised = max(0, autograder_wrong_total - autograder_wrong_revised)

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
            "autograder_wrong_human_correct": _case_stats(
                case_counts.get("autograder_wrong_human_correct", 0),
                revision_count=revision_count,
                total_evaluations=total_evaluations,
                autograder_wrong_total=autograder_wrong_total,
                include_autograder_share=True,
            ),
            "autograder_correct_human_wrong": _case_stats(
                case_counts.get("autograder_correct_human_wrong", 0),
                revision_count=revision_count,
                total_evaluations=total_evaluations,
                autograder_wrong_total=autograder_wrong_total,
                include_autograder_share=False,
            ),
            "both_wrong": _case_stats(
                case_counts.get("both_wrong", 0),
                revision_count=revision_count,
                total_evaluations=total_evaluations,
                autograder_wrong_total=autograder_wrong_total,
                include_autograder_share=True,
            ),
        },
        "autograder_wrong_breakdown": {
            "corrected": _breakdown_stats(
                corrected_revision_count,
                total_evaluations=total_evaluations,
                autograder_wrong_total=autograder_wrong_total,
            ),
            "not_revised": _breakdown_stats(
                autograder_wrong_unrevised,
                total_evaluations=total_evaluations,
                autograder_wrong_total=autograder_wrong_total,
            ),
            "revised_but_wrong": _breakdown_stats(
                both_wrong_count,
                total_evaluations=total_evaluations,
                autograder_wrong_total=autograder_wrong_total,
            ),
        },
        "breakdowns": {},
    }

    breakdowns: Dict[str, List[Dict[str, object]]] = {}
    for dimension_key, column_name in (
        ("ground_truth", columns.ground_truth),
        ("autograder_label", columns.autograder),
        ("human_label", columns.human),
    ):
        if not column_name or column_name not in normalized.columns:
            continue

        entries = _dimension_breakdowns(
            normalized=normalized,
            column_name=column_name,
            dimension_key=dimension_key,
            display_maps=display_maps,
            case_masks=case_masks,
            revisions=revisions,
            autograder_wrong_mask=autograder_wrong_mask,
            total_evaluations=total_evaluations,
        )
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
