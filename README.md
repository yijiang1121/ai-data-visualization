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

This writes `data/toy_grading_dataset.csv` and prints a small summary in the
terminal. The accuracy computation step only needs the CSV, so you can skip this
if the dataset already exists.

## 2. Compute accuracy metrics with pandas

Use the metrics module to calculate accuracy for both graders and serialize the
results to JSON:

```bash
python metrics/compute_accuracy.py \
  --input data/toy_grading_dataset.csv \
  --output public/accuracy.json \
  --revision-output public/revision.json
```

The script validates the dataset columns, normalizes label text, and reports
both overall and per-prompt accuracy. `accuracy.json` looks similar to:

```json
{
  "summary": {
    "total_records": 3000,
    "autograder_accuracy": 0.752,
    "human_accuracy": 0.7573
  },
  "per_prompt": [
    {
      "prompt_id": "PROMPT_0001",
      "count": 3,
      "autograder_accuracy": 0.6667,
      "human_accuracy": 1.0
    }
  ]
}
```

Revision-specific metrics are written to a sibling `revision.json`, which adds
precision and recall perspectives to the revision mix:

```json
{
  "overall": {
    "revision_count": 1217,
    "revision_rate": 0.4057,
    "correct_revision_precision": 0.4618,
    "autograder_wrong_recall": 0.7493
  },
  "cases": {
    "autograder_wrong_human_correct": {
      "count": 562,
      "share_of_revisions": 0.4618,
      "share_of_autograder_wrong": 0.7493
    }
  }
}
```

Re-run the command whenever the dataset changes to keep both JSON files current.

## 3. View the dashboard

Open `public/index.html` in a browser. The page fetches `accuracy.json` and
`revision.json` from the same directory, then renders the headline metrics,
revision precision/recall insights, and the prompt-level table. No build step or
framework is required—host the `public/` directory anywhere that serves static
files.

## Repository layout

```
.
├── data/                       # CSV datasets
├── metrics/                    # pandas-based metric utilities
├── public/                     # Static dashboard assets
├── generate_grading_dataset.py # Synthetic dataset generator (optional)
└── README.md
```
