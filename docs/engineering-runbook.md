# Pose annotation QA engineering runbook

This repository validates COCO keypoint schemas and spatial consistency, partitions
annotations for review, and reports retention. See the
[README input contract](../README.md#executable-validation-contract),
[QA command](../README.md#run-qa), and [retention command](../README.md#evaluate).
Temporal checks are planned; the existing temporal configuration is not executed.

## Local verification

Run from the repository root with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Dependency installation needs package-network access. Once installed, the test
suite runs on CPU with generated fixtures and does not download model weights or
datasets. On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell
and run `python -m pytest -q`.

Tests use generated annotations to check malformed schemas, nonfinite values,
visibility/count consistency, actual image dimensions, missing metadata,
per-annotation reasons, and reconciliation of accepted/flagged/rejected counts.
They do not measure QA precision or improvement on a labelled dataset.

## Data and artifact contract

- Each input COCO file needs `images` entries with integer IDs and positive width
  and height. Annotations must refer to that metadata. Visibility is categorical
  0/1/2, and `num_keypoints` counts points whose visibility is nonzero.
- Use distinct input and output directories. Keep a copy of source annotations;
  the pipeline writes `accepted.json`, `flagged.json`, `rejected.json` and
  `qa_report.json` with review reasons and preserved image/category metadata.
- Schema errors and missing image metadata are rejected. A spatial failure uses
  the configured flag/reject thresholds. Review thresholds for the dataset rather
  than treating them as universally valid anatomy rules.
- Retention counts only accepted annotations and reconciles all partitions. It
  is not precision: independently labelled errors are needed for precision/recall.
- Keep private imagery and annotations outside git. Review generated reports for
  source identifiers before sharing them; publish only permitted examples.

Human review remains necessary for ambiguous annotations and domain-specific
spatial judgments. No temporal or precision gain is claimed.
