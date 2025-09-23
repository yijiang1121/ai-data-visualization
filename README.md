# AI Data Visualization

A lightweight workflow for comparing the accuracy of an autograder and human
reviewers against a shared ground-truth dataset. Python (pandas) handles the
metric computation, while a static HTML dashboard presents the results.

## Prerequisites

- Python 3.10+
- [pip](https://pip.pypa.io/en/stable/). Install dependencies with `pip install -r requirements.txt`.

The optional dataset generator requires `numpy`, which is already included in the
requirements file.

## 1. Generate or refresh the dataset (optional)

If you would like to rebuild the synthetic grading dataset, run:

```bash
python generate_grading_dataset.py
```

This writes `data/toy_grading_dataset.csv`, prints a small summary in the
terminal, and removes the legacy confusion-matrix images if they are present.
The CSV includes three derived columns (`*_GNP`) that collapse the detailed
labels into the aggregated G/N/P buckets used later in the workflow. The
accuracy computation step only needs the CSV, so you can skip this if the
dataset already exists.

## 2. Compute accuracy metrics with pandas

Use the metrics module to calculate accuracy for both graders and serialize the
results to JSON:

```bash
python metrics/compute_accuracy.py \
  --input data/toy_grading_dataset.csv \
  --output public/accuracy.json \
  --revision-output public/revision.json
```

The script validates the required columns (`Prompt_ID`, `Human_Grade`,
`Auto_Grade`, `Ground_Truth`), normalizes label text, and recreates the
G/N/P-derived columns if they are missing. It then computes metrics for each
supported grading scale—the detailed four-point scale and the aggregated
G/N/P view. The resulting `accuracy.json` surfaces the default scale at the top
level while all metrics live under the `scales` key:

```json
{
  "default_scale": "four_level",
  "scale_labels": {
    "four_level": "4-Point Detail",
    "gnp": "G / N / P"
  },
  "scales": {
    "four_level": { "summary": { ... }, "per_prompt": [ ... ] },
    "gnp": { "summary": { ... }, "per_prompt": [ ... ] }
  }
}
```

Revision-specific metrics are written to a sibling `revision.json`, which adds
precision/recall details plus case breakdowns for every supported scale:

```json
{
  "default_scale": "four_level",
  "scale_labels": {
    "four_level": "4-Point Detail",
    "gnp": "G / N / P"
  },
  "overall": {
    "total_evaluations": 3000,
    "revision_count": 1217,
    "revision_rate": 0.4057,
    "correct_revision_count": 562,
    "correct_revision_precision": 0.4618,
    "autograder_wrong_total": 750,
    "corrected_autograder_wrong": 562,
    "autograder_wrong_recall": 0.7493,
    "mistake_repetition_factor": 3
  },
    "cases": {
      "autograder_wrong_human_correct": {
        "count": 562,
        "share_of_revisions": 0.4618,
        "share_of_total": 0.1873,
        "share_of_autograder_wrong": 0.7493
      },
      "autograder_correct_human_wrong": {
        "count": 532,
        "share_of_revisions": 0.4371,
        "share_of_total": 0.1773,
        "share_of_autograder_wrong": null
      },
      "both_correct": {
        "count": 0,
        "share_of_revisions": 0.0,
        "share_of_total": 0.0,
        "share_of_autograder_wrong": null
      },
      "both_wrong": {
        "count": 123,
        "share_of_revisions": 0.1011,
        "share_of_total": 0.041,
        "share_of_autograder_wrong": 0.164
    }
  },
  "autograder_wrong_breakdown": {
    "corrected": { "count": 562, ... },
    "not_revised": { "count": 65, ... },
    "revised_but_wrong": { "count": 123, ... }
  },
  "breakdowns": {
    "ground_truth": [ { "label": "highly satisfying", ... } ],
    "autograder_label": [ ... ],
    "human_label": [ ... ]
  },
  "scales": {
    "four_level": { ... },
    "gnp": { ... }
  }
}
```

Re-run the command whenever the dataset changes to keep both JSON files current.

## 3. View the dashboard

Open `public/index.html` in a browser. The page fetches `accuracy.json` and
`revision.json` from the same directory, then renders the headline metrics,
revision precision/recall insights, and the prompt-level table. Scale-aware
controls let you view the detailed four-level results or the aggregated G/N/P
perspective that the metrics script exports. No build step or framework is
required—host the `public/` directory anywhere that serves static files.

## Repository layout

```
.
├── data/                       # CSV datasets
├── metrics/                    # pandas-based metric utilities
├── public/                     # Static dashboard assets
├── generate_grading_dataset.py # Synthetic dataset generator (optional)
└── README.md
```
