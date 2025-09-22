"""Utilities for computing grading accuracy metrics."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

REQUIRED_COLUMNS = {"Prompt_ID", "Human_Grade", "Auto_Grade", "Ground_Truth"}
# Columns that should be auto-generated to expose the high level "G / N / P" view of
# the ratings.  Each derived column points to the original column from which its
# value should be mapped.
DERIVED_GNP_COLUMNS = {
    "Auto_Grade_GNP": "Auto_Grade",
    "Human_Grade_GNP": "Human_Grade",
    "Ground_Truth_GNP": "Ground_Truth",
}

# Mapping from the detailed satisfaction strings into their coarse-grained
# categories.  For example ``"slightly unsatisfying"`` falls into the ``"N"``
# bucket.
GNP_MAPPING = {
    "highly satisfying": "G",
    "slightly satisfying": "G",
    "slightly unsatisfying": "N",
    "highly unsatisfying": "P",
}

# Configuration describing each grading scale the script can report on.  The
# configuration keeps the column names together so the logic below can treat all
# scales uniformly.  Adding a new scale is as simple as adding another entry in
# this dictionary.
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

# Collect the set of all label related columns used by any scale.  This allows
# helper functions to normalise them without having to know which scale is
# currently being processed.
LABEL_COLUMNS = {
    column
    for config in SCALE_CONFIGS.values()
    for column in config["columns"].values()
}

# Order-preserving configuration describing the revision "cases" that appear in
# the JSON payload.  The boolean value indicates whether the case should report
# ``share_of_autograder_wrong``.
CASE_CONFIGS: Tuple[Tuple[str, bool], ...] = (
    ("autograder_wrong_human_correct", True),
    ("autograder_correct_human_wrong", False),
    ("both_wrong", True),
)

DIMENSION_FIELDS: Dict[str, Tuple[str, str]] = {
    "ground_truth": ("ground_truth", "ground_truth_display"),
    "autograder_label": ("autograder_label", "autograder_label_display"),
    "human_label": ("human_label", "human_label_display"),
}


@dataclass(frozen=True)
class ScaleColumns:
    """Convenience container for the expected columns of a grading scale."""

    ground_truth: str
    autograder: str
    human: str


def _normalize(series: pd.Series) -> pd.Series:
    """Normalize label text for reliable comparisons."""

    # Normalisation works by comparing the *text* of each label.  It strips the
    # whitespace and lower-cases the value so that labels such as
    # ``"Highly Satisfying"`` and ``"  highly satisfying  "`` are treated as the
    # same answer.
    return series.astype(str).str.strip().str.lower()


def derive_gnp_label(value: object) -> str | None:
    """Return the ``G``/``N``/``P`` bucket for a detailed satisfaction label.

    Examples
    --------
    >>> derive_gnp_label("highly unsatisfying")
    'P'
    >>> derive_gnp_label(" Slightly satisfying  ")
    'G'
    >>> derive_gnp_label(None) is None
    True
    """

    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    return GNP_MAPPING.get(text.lower())


def ensure_gnp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Populate the derived G/N/P columns based on the detailed ratings.

    The helper is intentionally side-effect friendly: it simply adds the new
    columns to ``df`` when the source columns exist and returns the modified
    frame.  Callers can therefore write ``df = ensure_gnp_columns(df)`` for
    clarity.
    """

    for derived_column, source_column in DERIVED_GNP_COLUMNS.items():
        if source_column not in df.columns:
            continue
        df[derived_column] = df[source_column].map(derive_gnp_label)

    return df


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the grading dataset and validate required columns.

    Rows missing any required column are dropped to avoid skewed metrics.  This
    mirrors the behaviour of a spreadsheet filter: only rows containing a full
    set of labels are included in the analysis.
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
    """Round values for presentation while preserving ``None`` values."""

    return None if value is None else round(value, 4)


def _format_rate(numerator: int, denominator: int) -> float | None:
    """Return a rounded ratio or ``None`` when the denominator is zero."""

    if not denominator:
        return None
    return _format(numerator / denominator)


def _safe_mean(series: pd.Series) -> float | None:
    """Return ``series.mean()`` while gracefully handling empty inputs."""

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
    """Map normalized label values back to their display text.

    The mapping allows the reporting code to normalise values for comparison
    while still displaying the original human readable labels.  For example, if
    the column contains ``" Highly Satisfying"`` and ``"highly satisfying"``
    they are both normalised to ``"highly satisfying"`` but we remember the
    neatly formatted version to display in the final JSON output.
    """

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

    # ``autograder_matches`` and ``human_matches`` are boolean Series telling us
    # for each row whether the assessor agreed with the ground truth label.
    autograder_matches = (
        normalized[columns.autograder] == normalized[columns.ground_truth]
    )
    human_matches = normalized[columns.human] == normalized[columns.ground_truth]

    total_evaluations = int(len(normalized))
    has_prompt_id = "Prompt_ID" in normalized.columns
    unique_prompts = (
        int(normalized["Prompt_ID"].nunique()) if has_prompt_id else total_evaluations
    )

    # When prompt identifiers are available, we only count one evaluation per
    # prompt for the autograder so that repeated evaluations do not overweight a
    # single prompt.  As a tiny example consider two prompts ``A`` and ``B``
    # where ``A`` is graded twice.  Using ``.first()`` means both prompts still
    # contribute exactly one data point to the aggregated accuracy.
    if has_prompt_id and unique_prompts:
        prompt_groups = normalized.groupby("Prompt_ID", sort=False)
        prompt_autograder = prompt_groups[columns.autograder].first()
        prompt_ground_truth = prompt_groups[columns.ground_truth].first()
        autograder_accuracy_value = _safe_mean(prompt_autograder == prompt_ground_truth)
    else:
        autograder_accuracy_value = _safe_mean(autograder_matches)

    # ``summary`` matches the structure used by the public JSON output.  Every
    # value is rounded where appropriate to keep the file readable.
    summary = {
        "total_evaluations": total_evaluations,
        "unique_prompts": unique_prompts,
        "autograder_accuracy": _format(autograder_accuracy_value),
        "autograder_evaluations": unique_prompts,
        "human_accuracy": _format(_safe_mean(human_matches)),
        "human_evaluations": total_evaluations,
    }

    per_prompt: List[Dict[str, object]] = []
    if has_prompt_id:
        for prompt_id, group in normalized.groupby("Prompt_ID", sort=True):
            # Within a prompt we treat the comparison as a simple mean of the
            # boolean matches.  ``_safe_mean`` returns ``None`` for empty groups,
            # which keeps the JSON concise for prompts without comparisons.
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

    # ``display_maps`` stores the prettiest version of each label so that the
    # JSON report can show human-friendly text while metrics operate on the
    # normalised values.
    display_maps = {
        column: _build_display_map(df[column])
        for column in LABEL_COLUMNS
        if column in df.columns
    }
    for column in LABEL_COLUMNS:
        if column in normalized.columns:
            normalized[column] = _normalize(normalized[column])

    accuracy_scales: Dict[str, Dict[str, object]] = {
        scale_key: _compute_accuracy_for_scale(normalized, config)
        for scale_key, config in SCALE_CONFIGS.items()
    }
    default_accuracy = accuracy_scales.get(
        DEFAULT_SCALE_KEY,
        next(iter(accuracy_scales.values()), {"summary": {}, "per_prompt": []}),
    )

    revision_scales: Dict[str, Dict[str, object]] = {
        scale_key: _compute_revision_metrics_for_scale(
            normalized=normalized,
            display_maps=display_maps,
            config=config,
        )
        for scale_key, config in SCALE_CONFIGS.items()
    }
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

    # ``autograder_wrong`` and ``human_wrong`` mark which rows contain mistakes.
    # Combining those with the ``revisions`` Series allows the caller to ask
    # specific questions such as "where did the human fix the autograder?".  For
    # example, if a row has ``autograder_wrong=True`` and ``revisions=True`` we
    # know the human wrote a different label for that prompt, making it a
    # candidate for a useful correction.
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

    # ``share_of_revisions`` answers "out of all the revisions, how many fall
    # into this scenario?".  ``share_of_total`` offers the same view but across
    # the full dataset rather than just revised responses.
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

    # These metrics are used in nested dictionaries and mirror the structure of
    # ``_case_stats`` so that the JSON output reads naturally.
    return {
        "count": count,
        "share_of_autograder_wrong": _format_rate(count, autograder_wrong_total),
        "share_of_total": _format_rate(count, total_evaluations),
    }


def _format_case_entries(
    case_counts: Dict[str, int],
    *,
    revision_count: int,
    total_evaluations: int,
    autograder_wrong_total: int,
) -> Dict[str, Dict[str, object]]:
    """Return ``_case_stats`` for every configured revision case."""

    return {
        case: _case_stats(
            case_counts.get(case, 0),
            revision_count=revision_count,
            total_evaluations=total_evaluations,
            autograder_wrong_total=autograder_wrong_total,
            include_autograder_share=include_autograder_share,
        )
        for case, include_autograder_share in CASE_CONFIGS
    }


def _format_autograder_breakdown(
    *,
    corrected: int,
    unrevised: int,
    revised_but_wrong: int,
    total_evaluations: int,
    autograder_wrong_total: int,
) -> Dict[str, Dict[str, object]]:
    """Return the standard breakdown for autograder mistakes."""

    return {
        "corrected": _breakdown_stats(
            corrected,
            total_evaluations=total_evaluations,
            autograder_wrong_total=autograder_wrong_total,
        ),
        "not_revised": _breakdown_stats(
            unrevised,
            total_evaluations=total_evaluations,
            autograder_wrong_total=autograder_wrong_total,
        ),
        "revised_but_wrong": _breakdown_stats(
            revised_but_wrong,
            total_evaluations=total_evaluations,
            autograder_wrong_total=autograder_wrong_total,
        ),
    }


def _split_autograder_mistakes(
    *, corrected: int, both_wrong: int, autograder_wrong_total: int
) -> Tuple[int, int]:
    """Return counts for revised and unrevised autograder mistakes."""

    autograder_wrong_revised = corrected + both_wrong
    autograder_wrong_unrevised = max(
        0, autograder_wrong_total - autograder_wrong_revised
    )
    return autograder_wrong_revised, autograder_wrong_unrevised


def _resolve_display_label(
    column: str, value: object, display_maps: Dict[str, Dict[str, str]]
) -> str:
    """Return a human-readable label for ``value`` in ``column``."""

    mapping = display_maps.get(column, {})
    if isinstance(value, str):
        return mapping.get(value, value.title())
    return mapping.get(value, str(value))


def _summarize_label_counts(
    indices: pd.Index,
    *,
    revisions: pd.Series,
    case_masks: Dict[str, pd.Series],
    autograder_wrong_mask: pd.Series,
) -> Dict[str, object]:
    """Collect frequently used counts for a single label slice."""

    revision_count = int(revisions.loc[indices].sum())
    case_counts = {
        case: int(mask.loc[indices].sum())
        for case, mask in case_masks.items()
    }
    autograder_wrong_total = int(autograder_wrong_mask.loc[indices].sum())

    corrected = case_counts.get("autograder_wrong_human_correct", 0)
    both_wrong = case_counts.get("both_wrong", 0)
    _, autograder_wrong_unrevised = _split_autograder_mistakes(
        corrected=corrected,
        both_wrong=both_wrong,
        autograder_wrong_total=autograder_wrong_total,
    )

    return {
        "revision_count": revision_count,
        "case_counts": case_counts,
        "autograder_wrong_total": autograder_wrong_total,
        "corrected": corrected,
        "both_wrong": both_wrong,
        "autograder_wrong_unrevised": autograder_wrong_unrevised,
    }


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

        # ``indices`` allows us to slice the boolean masks derived outside of
        # this helper.  Using ``loc`` ensures we line up by index rather than by
        # order which protects against accidental misalignment.
        indices = group.index
        label_counts = _summarize_label_counts(
            indices,
            revisions=revisions,
            case_masks=case_masks,
            autograder_wrong_mask=autograder_wrong_mask,
        )
        display_label = _resolve_display_label(column_name, label, display_maps)

        entry = {
            "total_evaluations": label_total,
            "revision_count": label_counts["revision_count"],
            "revision_rate": _format_rate(
                label_counts["revision_count"], label_total
            ),
            "correct_revision_count": label_counts["corrected"],
            "correct_revision_precision": _format_rate(
                label_counts["corrected"], label_counts["revision_count"]
            ),
            "autograder_wrong_total": label_counts["autograder_wrong_total"],
            "corrected_autograder_wrong": label_counts["corrected"],
            "autograder_wrong_recall": _format_rate(
                label_counts["corrected"], label_counts["autograder_wrong_total"]
            ),
            "cases": _format_case_entries(
                label_counts["case_counts"],
                revision_count=label_counts["revision_count"],
                total_evaluations=label_total,
                autograder_wrong_total=label_counts["autograder_wrong_total"],
            ),
            "autograder_wrong_breakdown": _format_autograder_breakdown(
                corrected=label_counts["corrected"],
                unrevised=label_counts["autograder_wrong_unrevised"],
                revised_but_wrong=label_counts["both_wrong"],
                total_evaluations=label_total,
                autograder_wrong_total=label_counts["autograder_wrong_total"],
            ),
            "label": label,
            "label_display": display_label,
        }

        field_names = DIMENSION_FIELDS.get(dimension_key)
        if field_names:
            label_field, display_field = field_names
            entry[label_field] = label
            entry[display_field] = display_label

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

    # ``revisions`` is ``True`` whenever the human and autograder disagree.
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

    # Convert the boolean Series into integer counts so the rest of the
    # function works with simple numbers.
    case_counts = {case: int(mask.sum()) for case, mask in case_masks.items()}
    revision_count = int(revisions.sum())
    autograder_wrong_total = int(autograder_wrong_mask.sum())
    corrected_revision_count = case_counts.get("autograder_wrong_human_correct", 0)
    both_wrong_count = case_counts.get("both_wrong", 0)
    _, autograder_wrong_unrevised = _split_autograder_mistakes(
        corrected=corrected_revision_count,
        both_wrong=both_wrong_count,
        autograder_wrong_total=autograder_wrong_total,
    )

    unique_repetitions = 1
    if "Repetition" in normalized.columns:
        # Some datasets contain a "Repetition" column to indicate how many
        # times a prompt was shown.  Tracking the number of unique repetitions
        # helps downstream consumers detect patterns such as "mistakes repeat
        # across three separate tries".
        unique_repetitions = max(
            1, int(normalized["Repetition"].dropna().nunique())
        )

    # ``revision`` mirrors the nested structure written to ``revision.json``.
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
        "cases": _format_case_entries(
            case_counts,
            revision_count=revision_count,
            total_evaluations=total_evaluations,
            autograder_wrong_total=autograder_wrong_total,
        ),
        "autograder_wrong_breakdown": _format_autograder_breakdown(
            corrected=corrected_revision_count,
            unrevised=autograder_wrong_unrevised,
            revised_but_wrong=both_wrong_count,
            total_evaluations=total_evaluations,
            autograder_wrong_total=autograder_wrong_total,
        ),
        "breakdowns": {},
    }

    # Build per-label breakdowns so the JSON can answer questions like "which
    # ground truth labels are most often revised?".
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
    """Serialize ``metrics`` to ``output_path`` as pretty-printed JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
        fh.write("\n")


def parse_args() -> argparse.Namespace:
    """Return parsed command-line arguments for the script."""

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
    """Entry point used by the ``__main__`` guard and the tests."""

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
