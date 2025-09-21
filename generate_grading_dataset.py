"""Generate a reproducible toy grading dataset.

The implementation mirrors the originally provided snippet while retaining the
improvements needed for convenient reuse:

* keep the pandas/numpy based data generation with explicit seeds
* persist the dataset to ``data/toy_grading_dataset.csv``
* print a concise textual summary of the resulting dataframe
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_PROMPTS = 1_000
N_REPETITIONS = 3
GRADING_SCALE = [
    "highly satisfying",
    "slightly satisfying",
    "slightly unsatisfying",
    "highly unsatisfying",
]
GRADER_ACCURACY = 0.75
RANDOM_SEED = 42
DATASET_PATH = Path("data/toy_grading_dataset.csv")
LEGACY_ARTIFACTS = [
    Path("data/auto_vs_ground_truth_confusion.png"),
    Path("data/human_vs_ground_truth_confusion.png"),
]


# ---------------------------------------------------------------------------
# Dataset generation helpers
# ---------------------------------------------------------------------------
def set_seeds(seed: int) -> None:
    """Seed NumPy and Python's ``random`` module for reproducibility."""

    np.random.seed(seed)
    random.seed(seed)


def build_ground_truth_labels(n_prompts: int, grading_scale: Iterable[str]) -> list[str]:
    """Create an evenly distributed, shuffled list of ground-truth labels."""

    grading_scale = list(grading_scale)
    prompts_per_category = n_prompts // len(grading_scale)
    remainder = n_prompts % len(grading_scale)

    labels: list[str] = []
    for idx, label in enumerate(grading_scale):
        count = prompts_per_category + (1 if idx < remainder else 0)
        labels.extend([label] * count)

    np.random.shuffle(labels)
    return labels


def sample_grade(ground_truth_label: str, grading_scale: Iterable[str], accuracy: float) -> str:
    """Return a label that matches the ground truth with probability ``accuracy``."""

    if np.random.random() < accuracy:
        return ground_truth_label

    choices = [label for label in grading_scale if label != ground_truth_label]
    return np.random.choice(choices)


def generate_toy_dataset(
    n_prompts: int = N_PROMPTS,
    n_repetitions: int = N_REPETITIONS,
    grading_scale: Iterable[str] = GRADING_SCALE,
    accuracy: float = GRADER_ACCURACY,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Replicate the original dataset generation logic using pandas."""

    set_seeds(random_seed)

    base_prompt_ids = [f"PROMPT_{idx + 1:04d}" for idx in range(n_prompts)]
    ground_truth = build_ground_truth_labels(n_prompts, grading_scale)

    auto_grades = [sample_grade(gt, grading_scale, accuracy) for gt in ground_truth]

    records: list[dict[str, object]] = []
    for prompt_idx in range(n_prompts):
        prompt_id = base_prompt_ids[prompt_idx]
        gt_label = ground_truth[prompt_idx]
        auto_label = auto_grades[prompt_idx]

        for repetition in range(1, n_repetitions + 1):
            human_label = sample_grade(gt_label, grading_scale, accuracy)
            records.append(
                {
                    "Prompt_ID": prompt_id,
                    "Repetition": repetition,
                    "Human_Grade": human_label,
                    "Auto_Grade": auto_label,
                    "Ground_Truth": gt_label,
                }
            )

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Persistence and reporting helpers
# ---------------------------------------------------------------------------
def save_dataset(df: pd.DataFrame, path: Path) -> Path:
    """Persist the dataset as CSV and return the resulting path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def summarize_dataset(df: pd.DataFrame) -> None:
    """Print dataset dimensions and a small preview."""

    print(f"Dataset shape: {df.shape}")
    print(f"Number of unique prompts: {df['Prompt_ID'].nunique()}")
    repetitions = df.groupby("Prompt_ID").size()
    print(f"Repetitions per prompt: {repetitions.iloc[0] if not repetitions.empty else 0}")
    print(df.head(10))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    df = generate_toy_dataset()
    csv_path = save_dataset(df, DATASET_PATH)
    print(f"Dataset saved to: {csv_path}")

    for artifact in LEGACY_ARTIFACTS:
        if artifact.exists():
            artifact.unlink()
            print(f"Removed legacy artifact: {artifact}")

    summarize_dataset(df)


if __name__ == "__main__":
    main()
