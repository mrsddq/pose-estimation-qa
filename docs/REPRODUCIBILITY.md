# Reproducibility Plan

## Environment

- Python: 3.10
- Dependencies: pinned in `requirements.txt`
- Config: `configs/qa.yaml`

## Data Contract

Each QA run should record annotation source/version, COCO schema version, image count, annotation count, reviewer labels, and checksum or DVC hash.

## Run Order

1. Validate raw COCO keypoint annotations.
2. Run spatial and temporal QA checks.
3. Save flagged/rejected annotations.
4. Evaluate against reviewer labels when available.
5. Generate visual review panels under `assets/`.

`outputs/metrics/smoke_test_results.csv` is a schema example only. It is not QA precision.
