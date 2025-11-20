# Pose Estimation Annotation QA

Professional annotation quality assurance system for pose estimation datasets. Built for agricultural and biological vision data. Achieved ~30% improvement in dataset precision through spatial and temporal consistency checks.

> This repository contains anonymised workflow code. No client imagery or proprietary data is included.

## Impact

| Metric | Value |
|---|---|
| Dataset precision improvement | ~30% |
| QA dimensions | Spatial accuracy, temporal consistency, schema compliance |
| Domain | Agricultural / biological imagery |
| Annotation format | COCO keypoints |

## QA Pipeline

```
Raw annotations (COCO keypoints JSON)
  └─ Schema validation        checks required fields, keypoint count, bbox presence
       └─ Spatial consistency  flags anatomically implausible keypoint positions
            └─ Temporal consistency  checks keypoint movement between adjacent frames
                 └─ Confidence filtering  removes low-confidence keypoints below threshold
                      └─ QA decision: accept / flag for review / auto-reject
```

## Quickstart

```bash
git clone https://github.com/your-username/pose-estimation-qa
cd pose-estimation-qa
pip install -r requirements.txt
```

## Data Format

Input: COCO-format keypoints JSON.

```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 42,
      "keypoints": [x1, y1, v1, x2, y2, v2, ...],
      "num_keypoints": 17,
      "bbox": [x, y, w, h],
      "score": 0.91
    }
  ]
}
```

Place data as:
```
data/
  raw_annotations/    ← original model output JSON files
  qa_output/          ← QA-filtered annotation JSON files
```

## Running QA

```bash
# Run full QA pipeline
python scripts/qa/run_qa.py --input data/raw_annotations/ --output data/qa_output/ --config configs/qa.yaml

# Evaluate precision improvement
python scripts/evaluation/evaluate_precision.py --before data/raw_annotations/ --after data/qa_output/

# Visualise flagged cases
python scripts/visualize/visualize_flags.py --annotations data/qa_output/flagged.json
```

## Sample Outputs

| File | Contents |
|---|---|
| `assets/01_qa_workflow.png` | Anonymised QA pipeline flowchart |
| `assets/02_before_after_qa.png` | Pose skeleton: noisy vs corrected keypoints |
| `assets/03_precision_summary.png` | Before/after precision improvement chart |

## Configuration

See `configs/qa.yaml` for all QA thresholds — spatial distance limits, temporal velocity caps, confidence cutoffs, and schema rules.

## Limitations

- Spatial plausibility checks are heuristic and domain-specific — thresholds tuned for agricultural subjects
- Temporal checks require sequential frame ordering; shuffled datasets need sorting first
- No client data included — all examples use public COCO person annotations

## Environment

```
Python 3.10
numpy==1.26.0
opencv-python==4.8.1.78
pycocotools==2.0.7
matplotlib==3.8.0
```
