"""Unit tests for the grading metrics helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.compute_accuracy import (
    DEFAULT_SCALE_KEY,
    SCALE_CONFIGS,
    _build_display_map,
    _compute_accuracy_for_scale,
    _compute_revision_metrics_for_scale,
    _normalize,
    compute_metrics,
    derive_gnp_label,
    ensure_gnp_columns,
    load_dataset,
    write_metrics,
)


def test_normalize_lowercases_and_strips_whitespace() -> None:
    series = pd.Series(["  Highly Satisfying  ", "SLIGHTLY UNSATISFYING", None])

    normalized = _normalize(series)

    assert normalized.tolist() == [
        "highly satisfying",
        "slightly unsatisfying",
        "none",
    ]


def test_derive_gnp_label_maps_known_values_and_handles_missing() -> None:
    assert derive_gnp_label("Highly Satisfying") == "G"
    assert derive_gnp_label("slightly unsatisfying") == "N"
    assert derive_gnp_label("HIGHLY UNSATISFYING") == "P"
    assert derive_gnp_label("   ") is None
    assert derive_gnp_label(None) is None


def test_ensure_gnp_columns_adds_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "Prompt_ID": [1, 2],
            "Ground_Truth": ["Highly Satisfying", "Highly Unsatisfying"],
            "Auto_Grade": ["Slightly Satisfying", "Highly Unsatisfying"],
            "Human_Grade": ["Highly Satisfying", "Slightly Unsatisfying"],
        }
    )

    result = ensure_gnp_columns(df.copy())

    assert set(result.columns).issuperset(
        {"Ground_Truth_GNP", "Auto_Grade_GNP", "Human_Grade_GNP"}
    )
    assert result["Ground_Truth_GNP"].tolist() == ["G", "P"]
    assert result["Auto_Grade_GNP"].tolist() == ["G", "P"]
    assert result["Human_Grade_GNP"].tolist() == ["G", "N"]


def test_load_dataset_validates_required_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "grading.csv"
    pd.DataFrame(
        {
            "Prompt_ID": [1],
            "Human_Grade": ["Highly Satisfying"],
            "Ground_Truth": ["Highly Satisfying"],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_dataset(csv_path)


def test_load_dataset_drops_incomplete_rows_and_derives_gnp(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv_path = tmp_path / "grading.csv"
    pd.DataFrame(
        {
            "Prompt_ID": [1, 2, 3],
            "Human_Grade": [
                "Highly Satisfying",
                "Highly Unsatisfying",
                None,
            ],
            "Auto_Grade": [
                "Slightly Satisfying",
                "Highly Unsatisfying",
                "Highly Satisfying",
            ],
            "Ground_Truth": [
                "Highly Satisfying",
                "Highly Unsatisfying",
                "Highly Satisfying",
            ],
        }
    ).to_csv(csv_path, index=False)

    df = load_dataset(csv_path)
    captured = capsys.readouterr()

    assert len(df) == 2
    assert "Dropped 1 rows" in captured.out
    assert df["Auto_Grade_GNP"].tolist() == ["G", "P"]
    assert df["Human_Grade_GNP"].tolist() == ["G", "P"]


def test_compute_accuracy_for_scale_handles_prompt_grouping() -> None:
    normalized = pd.DataFrame(
        {
            "Prompt_ID": [101, 101, 202],
            "Ground_Truth": [
                "highly satisfying",
                "slightly unsatisfying",
                "highly unsatisfying",
            ],
            "Auto_Grade": [
                "slightly unsatisfying",
                "slightly unsatisfying",
                "slightly unsatisfying",
            ],
            "Human_Grade": [
                "highly satisfying",
                "highly unsatisfying",
                "highly satisfying",
            ],
        }
    )

    result = _compute_accuracy_for_scale(normalized, SCALE_CONFIGS["four_level"])

    assert result["summary"] == {
        "total_evaluations": 3,
        "unique_prompts": 2,
        "autograder_accuracy": 0.0,
        "autograder_evaluations": 2,
        "human_accuracy": 0.3333,
        "human_evaluations": 3,
    }
    assert result["per_prompt"] == [
        {
            "prompt_id": "101",
            "count": 2,
            "autograder_accuracy": 0.5,
            "human_accuracy": 0.5,
        },
        {
            "prompt_id": "202",
            "count": 1,
            "autograder_accuracy": 0.0,
            "human_accuracy": 0.0,
        },
    ]


def _build_sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Prompt_ID": [101, 101, 202],
            "Repetition": [1, 2, 1],
            "Ground_Truth": [
                "Highly Satisfying",
                "Slightly Unsatisfying",
                "Highly Unsatisfying",
            ],
            "Auto_Grade": [
                "Slightly Unsatisfying",
                "Slightly Unsatisfying",
                "Slightly Unsatisfying",
            ],
            "Human_Grade": [
                "Highly Satisfying",
                "Highly Unsatisfying",
                "Highly Satisfying",
            ],
        }
    )


def test_compute_metrics_returns_expected_structure() -> None:
    df = _build_sample_dataset()

    accuracy_metrics, revision_metrics = compute_metrics(df)

    assert accuracy_metrics["default_scale"] == DEFAULT_SCALE_KEY
    assert accuracy_metrics["summary"] == {
        "total_evaluations": 3,
        "unique_prompts": 2,
        "autograder_accuracy": 0.0,
        "autograder_evaluations": 2,
        "human_accuracy": 0.3333,
        "human_evaluations": 3,
    }
    assert accuracy_metrics["per_prompt"] == [
        {
            "prompt_id": "101",
            "count": 2,
            "autograder_accuracy": 0.5,
            "human_accuracy": 0.5,
        },
        {
            "prompt_id": "202",
            "count": 1,
            "autograder_accuracy": 0.0,
            "human_accuracy": 0.0,
        },
    ]
    assert accuracy_metrics["scale_labels"]["gnp"] == "G / N / P"
    assert (
        accuracy_metrics["scales"]["gnp"]["summary"]["total_evaluations"]
        == 3
    )

    assert revision_metrics["default_scale"] == DEFAULT_SCALE_KEY
    assert revision_metrics["overall"]["total_evaluations"] == 3
    assert revision_metrics["overall"]["revision_rate"] == 1.0
    assert revision_metrics["overall"]["mistake_repetition_factor"] == 2
    assert revision_metrics["cases"]["autograder_wrong_human_correct"]["count"] == 1
    assert revision_metrics["cases"]["autograder_correct_human_wrong"]["count"] == 1
    assert revision_metrics["cases"]["both_correct"]["count"] == 0
    assert revision_metrics["cases"]["both_wrong"]["count"] == 1
    assert (
        revision_metrics["autograder_wrong_breakdown"]["not_revised"]["count"]
        == 0
    )
    assert "agreement" in revision_metrics
    assert revision_metrics["agreement"]["overall"]["agreement_count"] == 0
    assert revision_metrics["slices"]["agreement"]["overall"]["agreement_count"] == 0
    assert (
        revision_metrics["slices"]["disagreement"]["overall"]["revision_count"]
        == 3
    )
    assert revision_metrics["overall"]["autograder_correct_total"] == 1
    assert revision_metrics["overall"]["autograder_correct_rate"] == 0.3333
    assert (
        revision_metrics["autograder_correct"]["overall"]["autograder_correct_total"]
        == 1
    )
    assert (
        revision_metrics["slices"]["autograder_correct"]["overall"][
            "autograder_correct_total"
        ]
        == 1
    )
    assert (
        revision_metrics["autograder_correct"]["cases"][
            "autograder_correct_human_wrong"
        ]["count"]
        == 1
    )
    assert (
        revision_metrics["autograder_correct"]["cases"]["both_correct"]["count"]
        == 0
    )


def test_compute_revision_metrics_for_scale_provides_breakdowns() -> None:
    df = ensure_gnp_columns(_build_sample_dataset())
    display_maps = {
        column: _build_display_map(df[column])
        for column in df.columns
        if column in SCALE_CONFIGS["four_level"]["columns"].values()
        or column.endswith("_GNP")
    }
    normalized = df.copy()
    for column in normalized.columns:
        if normalized[column].dtype == object:
            normalized[column] = _normalize(normalized[column])

    revision = _compute_revision_metrics_for_scale(
        normalized=normalized,
        display_maps=display_maps,
        config=SCALE_CONFIGS["four_level"],
    )

    overall = revision["overall"]
    assert overall["total_evaluations"] == 3
    assert overall["revision_count"] == 3
    assert overall["correct_revision_precision"] == 0.3333
    assert overall["autograder_wrong_total"] == 2
    assert overall["autograder_wrong_recall"] == 0.5
    assert overall["autograder_correct_total"] == 1
    assert overall["autograder_correct_rate"] == 0.3333

    cases = revision["cases"]
    assert cases["autograder_wrong_human_correct"]["count"] == 1
    assert cases["autograder_wrong_human_correct"]["share_of_autograder_wrong"] == 0.5
    assert cases["autograder_correct_human_wrong"]["count"] == 1
    assert cases["both_correct"]["count"] == 0
    assert cases["both_wrong"]["count"] == 1

    breakdown = revision["autograder_wrong_breakdown"]
    assert breakdown["corrected"]["count"] == 1
    assert breakdown["not_revised"]["count"] == 0
    assert breakdown["revised_but_wrong"]["count"] == 1

    ground_truth_breakdowns = revision["breakdowns"]["ground_truth"]
    labels = {entry["label"]: entry for entry in ground_truth_breakdowns}
    assert labels["highly satisfying"]["label_display"] == "Highly Satisfying"
    assert labels["highly satisfying"]["autograder_wrong_recall"] == 1.0
    assert revision["by_ground_truth"] == ground_truth_breakdowns

    autograder_correct = revision["autograder_correct"]
    assert (
        autograder_correct["cases"]["autograder_correct_human_wrong"]["count"]
        == 1
    )
    assert autograder_correct["cases"]["both_correct"]["count"] == 0
    correct_by_gt = {
        entry["label"]: entry
        for entry in autograder_correct["breakdowns"].get("ground_truth", [])
    }
    assert (
        correct_by_gt["slightly unsatisfying"]["autograder_correct_total"] == 1
    )

    agreement = revision["agreement"]
    assert agreement["overall"]["agreement_count"] == 0
    assert agreement["cases"]["both_correct"]["count"] == 0
    gt_agreement = {
        entry["label"]: entry
        for entry in agreement["breakdowns"].get("ground_truth", [])
    }
    assert gt_agreement["highly satisfying"]["agreement_count"] == 0


def test_agreement_slice_captures_alignment_metrics() -> None:
    df = pd.DataFrame(
        {
            "Prompt_ID": [1, 2, 3, 4],
            "Ground_Truth": [
                "Highly Satisfying",
                "Highly Unsatisfying",
                "Slightly Satisfying",
                "Slightly Unsatisfying",
            ],
            "Auto_Grade": [
                "Highly Satisfying",
                "Highly Unsatisfying",
                "Slightly Satisfying",
                "Highly Unsatisfying",
            ],
            "Human_Grade": [
                "Highly Satisfying",
                "Highly Unsatisfying",
                "Slightly Satisfying",
                "Highly Unsatisfying",
            ],
        }
    )

    df = ensure_gnp_columns(df)
    display_maps = {
        column: _build_display_map(df[column])
        for column in df.columns
        if column in SCALE_CONFIGS["four_level"]["columns"].values()
        or column.endswith("_GNP")
    }
    normalized = df.copy()
    for column in normalized.columns:
        if normalized[column].dtype == object:
            normalized[column] = _normalize(normalized[column])

    revision = _compute_revision_metrics_for_scale(
        normalized=normalized,
        display_maps=display_maps,
        config=SCALE_CONFIGS["four_level"],
    )
    agreement = revision["agreement"]

    overall = agreement["overall"]
    assert overall["agreement_count"] == 4
    assert overall["agreement_rate"] == 1.0
    assert overall["agreement_accuracy"] == 0.75

    cases = agreement["cases"]
    assert cases["both_correct"]["count"] == 3
    assert cases["both_wrong"]["count"] == 1
    assert cases["both_wrong"]["share_of_agreements"] == 0.25

    ground_truth_entries = {
        entry["label"]: entry
        for entry in agreement["breakdowns"].get("ground_truth", [])
    }
    assert ground_truth_entries["highly satisfying"]["agreement_accuracy"] == 1.0
    assert ground_truth_entries["slightly unsatisfying"]["agreement_accuracy"] == 0.0


def test_write_metrics_creates_expected_json(tmp_path: Path) -> None:
    output_path = tmp_path / "metrics.json"
    payload = {"summary": {"total": 3}}

    write_metrics(payload, output_path)

    assert output_path.exists()
    with output_path.open() as fh:
        data = json.load(fh)
    assert payload == {"summary": {"total": 3}}
    assert data["summary"] == payload["summary"]
    assert "generated_at" in data
    # ``fromisoformat`` raises ``ValueError`` if the timestamp is not valid ISO-8601.
    datetime.fromisoformat(data["generated_at"])
