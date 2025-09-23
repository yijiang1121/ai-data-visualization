"""Generate the streamlined dashboard payload for index_v2.html."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from compute_accuracy import compute_metrics, load_dataset

CASE_ORDER = [
    "autograder_wrong_human_correct",
    "autograder_correct_human_wrong",
    "both_correct",
    "both_wrong",
]

CASE_LABELS = {
    "autograder_wrong_human_correct": "Auto wrong, human correct",
    "autograder_correct_human_wrong": "Auto correct, human wrong",
    "both_correct": "Both correct",
    "both_wrong": "Both wrong",
}

DISAGREEMENT_SHARE_FIELD = "share_of_revisions"
AGREEMENT_SHARE_FIELD = "share_of_agreements"


def _format_share(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(value)


def _format_count(value) -> int:
    return int(value) if value is not None else 0


def build_outcomes(cases: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    outcomes: List[Dict[str, object]] = []
    for key in CASE_ORDER:
        case = cases.get(key, {}) or {}
        share = case.get(DISAGREEMENT_SHARE_FIELD)
        outcomes.append(
            {
                "key": key,
                "label": CASE_LABELS[key],
                "count": _format_count(case.get("count")),
                "share": _format_share(share),
            }
        )
    return outcomes


def build_focus_row(slice_key: str, slice_payload: Dict[str, object]) -> Dict[str, object]:
    overall = slice_payload.get("overall", {}) or {}
    cases = slice_payload.get("cases", {}) or {}

    if slice_key == "disagreement":
        rate = overall.get("revision_rate")
        count = overall.get("revision_count")
        share_field = DISAGREEMENT_SHARE_FIELD
        share_label = "of revisions"
    elif slice_key == "agreement":
        rate = overall.get("agreement_rate")
        count = overall.get("agreement_count")
        share_field = AGREEMENT_SHARE_FIELD
        share_label = "of agreements"
    else:
        return {}

    total_evaluations = overall.get("total_evaluations")

    case_entries: List[Dict[str, object]] = []
    for key in CASE_ORDER:
        case = cases.get(key, {}) or {}
        case_entries.append(
            {
                "key": key,
                "label": CASE_LABELS[key],
                "count": _format_count(case.get("count")),
                "share": _format_share(case.get(share_field)),
                "share_label": share_label,
            }
        )

    return {
        "key": slice_key,
        "label": slice_payload.get("label", slice_key.replace("_", " ").title()),
        "rate": rate,
        "revisions": {
            "count": _format_count(count),
            "total": _format_count(total_evaluations),
        },
        "cases": case_entries,
    }


def build_dashboard_payload(accuracy_metrics: Dict[str, object], revision_metrics: Dict[str, object]) -> Dict[str, object]:
    scales_payload: Dict[str, Dict[str, object]] = {}

    for scale_key, accuracy_data in (accuracy_metrics.get("scales") or {}).items():
        revision_data = (revision_metrics.get("scales") or {}).get(scale_key, {})
        if not accuracy_data or not revision_data:
            continue

        summary = accuracy_data.get("summary", {}) or {}
        overall = revision_data.get("overall", {}) or {}
        cases = revision_data.get("cases", {}) or {}
        slices = revision_data.get("slices", {}) or {}

        disagreement = slices.get("disagreement", {})
        agreement = slices.get("agreement", {})

        focus_rows = []
        if disagreement:
            focus_rows.append(build_focus_row("disagreement", disagreement))
        if agreement:
            focus_rows.append(build_focus_row("agreement", agreement))

        scales_payload[scale_key] = {
            "accuracy": {
                "total_evaluations": _format_count(summary.get("total_evaluations")),
                "unique_prompts": _format_count(summary.get("unique_prompts")),
                "autograder_accuracy": summary.get("autograder_accuracy"),
                "autograder_evaluations": _format_count(summary.get("autograder_evaluations")),
                "human_accuracy": summary.get("human_accuracy"),
                "human_evaluations": _format_count(summary.get("human_evaluations")),
            },
            "revision": {
                "rate": {
                    "value": overall.get("revision_rate"),
                    "revisions": _format_count(overall.get("revision_count")),
                    "total_reviews": _format_count(overall.get("total_evaluations")),
                },
                "precision": {
                    "value": overall.get("correct_revision_precision"),
                    "correct_revisions": _format_count(overall.get("correct_revision_count")),
                    "total_revisions": _format_count(overall.get("revision_count")),
                },
                "outcomes": build_outcomes(cases),
                "focus_rows": focus_rows,
            },
        }

    payload = {
        "default_scale": accuracy_metrics.get("default_scale"),
        "scale_labels": accuracy_metrics.get("scale_labels", {}),
        "scales": scales_payload,
    }
    payload["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["source"] = "dashboard_v2"
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/toy_grading_dataset.csv"),
        help="Path to the grading dataset CSV (default: data/toy_grading_dataset.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("public/dashboard_v2.json"),
        help="Where to write the streamlined dashboard payload (default: public/dashboard_v2.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)
    accuracy_metrics, revision_metrics = compute_metrics(df)
    payload = build_dashboard_payload(accuracy_metrics, revision_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"Wrote streamlined dashboard metrics to {args.output}")


if __name__ == "__main__":
    main()
