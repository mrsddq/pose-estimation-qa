# Pose Estimation Annotation QA

Portfolio-ready quality assurance toolkit for pose-estimation annotation datasets.

The repository focuses on COCO-keypoint validation, spatial consistency checks, temporal consistency checks, and review workflows. It does not include private imagery or unsupported precision-improvement claims.

## Highlights

- COCO keypoint schema validation
- Spatial plausibility checks
- Temporal movement checks for ordered frames
- Visualization script for flagged cases
- Evaluation workflow for before/after QA comparison

## Structure

```text
configs/
  qa.yaml
scripts/
  qa/run_qa.py
  evaluation/evaluate_precision.py
  visualize/visualize_flags.py
```

## Input Format

COCO-style keypoint annotations:

```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 42,
      "keypoints": [x1, y1, v1],
      "num_keypoints": 1,
      "bbox": [x, y, w, h],
      "score": 0.91
    }
  ]
}
```

## Run QA

```bash
python -m scripts.qa.run_qa --input data/raw_annotations --output data/qa_output --config configs/qa.yaml
```

## Evaluate

```bash
python -m scripts.evaluation.evaluate_precision --before data/raw_annotations --after data/qa_output
```

## Visualize

```bash
python -m scripts.visualize.visualize_flags --annotations data/qa_output/flagged.json
```

## Results

No private or verified public QA metrics are committed. Use [docs/QA_REPORT_TEMPLATE.md](docs/QA_REPORT_TEMPLATE.md) to document precision, recall, and reviewer workload changes after running on a labelled dataset.

## Limitations

- Spatial thresholds are domain-specific.
- Temporal checks require correctly ordered frames.
- Human review remains necessary for ambiguous cases.
